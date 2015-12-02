package com.sdl.smert

import scala.collection.mutable.Buffer
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import com.sdl.NBest._
import java.io.File
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import breeze.linalg._
import breeze.numerics._
import com.sdl.BleuStats
import scala.annotation.tailrec
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import breeze.stats.distributions.Gaussian
import breeze.stats.distributions.RandBasis
import org.apache.commons.math3.random.RandomGenerator
import org.apache.commons.math3.random.MersenneTwister

object SMERT {

  def doSvd(): Unit = {
    /*    val trainingSet : RDD[org.apache.spark.mllib.linalg.Vector] = for(nbest <- nbests; hyp <- nbest) yield{
      Vectors.dense(hyp.fVec.toArray)
    }
    trainin
    val trainingSetMat = new RowMatrix(trainingSet)
    val svd = trainingSetMat.computeSVD(, computeU = true)*/

  }

  def generateDirections(r : RandBasis, d : Int, noOfRandom : Int) = {
     val axes = DenseMatrix.eye[Float](d)
     val rand = DenseMatrix.rand(noOfRandom, d, Gaussian(0, 1)(r)).map(_.toFloat)
     DenseMatrix.vertcat(axes, rand)
  }
  
  def nbestToMatrix(in: NBest): DenseMatrix[Float] = {
    require(in.size > 0, "NBest needs to have at least one element")
    val fVecDim = in(0).fVec.size
    val buf = new ArrayBuffer[Float]
    for (hyp <- in) {
      require(hyp.fVec.size == fVecDim, "Feature vecs must be of the same dimension")
      buf ++= hyp.fVec.map(_.toFloat * -1.0f)
    }
    new DenseMatrix(fVecDim, in.size, buf.toArray)
  }

  @tailrec
  def applyError(prevErr: BleuStats, intervals: Seq[(Float, _, BleuStats)], accum: Buffer[(Float, (Double, Double), BleuStats)]): Unit = {
    intervals match {
      case Nil => Unit
      case head +: tail => {
        val err = if (head._1 != Float.NegativeInfinity) prevErr + head._3 else prevErr
        val res = (head._1, err.computeBleu(), err)
        accum += res
        applyError(err, tail, accum)
      }
    }
  }

  def iteration(nbests: Broadcast[Seq[Tuple2[DenseMatrix[Float], IndexedSeq[BleuStats]]]],
      indices: RDD[Int], point: DenseVector[Float], directions : DenseMatrix[Float]) = {
    val swept = indices.map(nbests.value(_)).map(Sweep.sweep(point, directions)_)
    val reduced = swept.reduce((a, b) => {
      for ((seqA, seqB) <- a zip b) yield {
        (seqA ++ seqB).toIndexedSeq
      }
    })
    val updates = for ((collected, d) <- reduced.view.zipWithIndex) yield {
      val sortedLine = collected.sortBy(interval => interval._1)
      var error = BleuStats.empty
      val intervals = for (intervalBoundary <- sortedLine) yield {
        val isInf = intervalBoundary._1 == Float.NegativeInfinity
        val bs = if (isInf) intervalBoundary._2 else intervalBoundary._3
        error = error + bs
        if (isInf)
          (intervalBoundary._1, bs.computeBleu(), bs)
        else
          (intervalBoundary._1, error.computeBleu(), error)
      }
      //for(i <- intervals) println(i)
      val max = intervals.sliding(2).maxBy(interval => if (interval(0)._1 == Float.NegativeInfinity) Double.MinValue else interval(1)._2._1)
      val best = max(1)
      val (bleu, bp) = best._2
      val prev = max(0)
      val update = (best._1 - prev._1) / 2
      println(f"Direction $d: Update at $update%s with BLEU: $bleu%.6f [$bp%.4f]")
      val updatedPoint = DenseVector(update, 1).t * Sweep.createProjectionMatrix(directions(d, ::).t, point)
      (updatedPoint, (bleu, bp))
    }
    updates.maxBy(_._2)
  }

  @tailrec
  def iterate(sc: SparkContext, prevPoint: DenseVector[Float], prevBleu: (Double, Double), nbests: Broadcast[Seq[Tuple2[DenseMatrix[Float], IndexedSeq[BleuStats]]]],
      indices : RDD[Int], deltaBleu: Double, r : RandBasis, noOfRandom : Int): (DenseVector[Float], (Double, Double)) = {
    val directions = generateDirections(r, prevPoint.length, noOfRandom)
    val (point, (bleu, bp)) = iteration(nbests, indices, prevPoint, directions)
    if ((bleu - prevBleu._1) < deltaBleu) 
      (prevPoint, prevBleu)
    else 
      iterate(sc, point.t, (bleu, bp), nbests, indices, deltaBleu, r, noOfRandom)    
  }

  def main(args: Array[String]): Unit = {
    case class Config(
      nbestsDir: File = new File("."),
      initialPoint: DenseVector[Float] = DenseVector(),
      localThreads: Option[Int] = None,
      noOfPartitions: Int = 100,
      deltaBleu: Double = 1.0E-6,
      random: RandomGenerator = new MersenneTwister(11l),
      noOfInitials: Int = 0,
      noOfRandom: Int = 250,
      out: File = new File("./params"))
    val parser = new scopt.OptionParser[Config]("smert") {
      head("smert", "1.0")
      //NBest Option
      opt[File]('n', "nbest_dir") required () action { (x, c) =>
        c.copy(nbestsDir = x)
      } validate (x => if (x.isDirectory()) success else failure("Nbest directory is not a directory")) text ("NBest input directory")
      // Initial point
      opt[String]('i', "initial") required () action { (x, c) =>
        c.copy(initialPoint = DenseVector(x.split(",").map(_.toFloat)))
      } text ("Initital starting point")
      opt[Int]('l', "local_threads") action { (x, c) =>
        c.copy(localThreads = Some(x))
      } text ("If supplied, run in a local master with this number of threads")
      opt[Int]('p', "partitions") action { (x, c) =>
        c.copy(noOfPartitions = x)
      } text ("Number of partitions")
      opt[String]('d', "delta_bleu") action { (x, c) =>
        c.copy(deltaBleu = x.toDouble)
      } text ("Termination condition for the BLEU score")
      opt[Int]('r', "random_seed") action { (x, c) =>
        c.copy(random = new MersenneTwister(x))
      } text ("Random Seed")
      opt[File]('o', "output") required () action { (x, c) =>
        c.copy(out = x)
      }
    }
    val cliConf = parser.parse(args, Config()).getOrElse(sys.exit())
    println(cliConf)
    import cliConf._
    val conf = new SparkConf().setAppName("SMERT")
    val rb = new RandBasis(random)
    for (l <- localThreads) conf.setMaster(s"local[$l]")
    val sc = new SparkContext(conf)
    val nbests = loadUCamNBest(nbestsDir).par.map {
      n =>
        {
          val mat = nbestToMatrix(n)
          val bs = n map (_.bs)
          (mat, bs)
        }
    }.seq
    val indices = sc.parallelize(0 until nbests.length)
    val nbestsBroadcast = sc.broadcast(nbests)
    val initials = scala.collection.immutable.Vector(initialPoint) ++ 
      (for (i <- 0 until noOfInitials) yield DenseVector.rand(initialPoint.size, Gaussian(0,1)(rb)).map(_.toFloat))
    val res = for (i <- initials) yield iterate(sc, i, (0.0, 0.0), nbestsBroadcast, indices, deltaBleu, rb, noOfRandom)
    val (finalPoint, (finalBleu, finalBP)) = res.maxBy(_._2._1)
    println(f"Found final point with BLEU $finalBleu%.6f [$finalBP%.4f]!")
    breeze.linalg.csvwrite(out, finalPoint.toDenseMatrix.map(_.toDouble))
    println(finalPoint)
  }
}
