package com.sdl.smert

import org.apache.commons.math3.random.JDKRandomGenerator
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
import org.apache.commons.math3.random.JDKRandomGenerator
import com.sdl.smert.Sweep._
import org.apache.spark.storage.StorageLevel

object SMERT {

  type BreezeNBest = (DenseMatrix[Float], IndexedSeq[BleuStats])

  def computeScore(nbests: RDD[BreezeNBest], params: DenseVector[Float]) = nbests.map {
    case (proj, bs) => {
      val scores = (params.t * proj).t
      bs(argsort(scores).last)
    }
  }.reduce(_ + _)

  def getGenerator(seed: Int = 11) = {
    val r = new JDKRandomGenerator()
    r.setSeed(seed.toLong)
    r
  }

  case class Config(
    nbestsDir: File = new File("."),
    nbestsObjDir: Option[String] = None,
    initialPoint: DenseVector[Float] = DenseVector(),
    localThreads: Option[Int] = None,
    noOfPartitions: Int = 100,
    deltaBleu: Double = 1.0E-6,
    random: RandomGenerator = getGenerator(),
    noOfInitials: Int = 50,
    noOfRandom: Int = 100,
    out: File = new File("./params"),
    affineDims: Set[Int] = Set.empty,
    verify: Boolean = false,
    activationFactor: Option[Double] = None,
    sweepFunc: SweepFunc = Sweep.sweepLine)

  def generateDirections(r: RandBasis, d: Int, noOfRandom: Int, affineDims: Set[Int]) = {
    val axes = for (i <- 0 until d) yield {
      if (!affineDims(i)) {
        val row = DenseMatrix.zeros[Float](1, d)
        row(0, i) = 1f
        Some(row)
      } else None
    }
    val axesMat = DenseMatrix.vertcat(axes.flatten: _*)
    val rand = DenseMatrix.rand(noOfRandom, d, Gaussian(0, 1)(r)).map(_.toFloat)
    affineDims map { c =>
      for (r <- 0 until noOfRandom) rand(r, c) = 0f
    }
    DenseMatrix.vertcat(axesMat, rand)
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

  def iteration(
    nbests: RDD[BreezeNBest],
    point: DenseVector[Float],
    directions: DenseMatrix[Float],
    activationFactor: Option[Double],
    sweepFunc: SweepFunc,
    validationSet: Option[RDD[BreezeNBest]]) = {
    val swept = nbests.map(Sweep.sweep(point, directions, sweepFunc)_)
    val reduced = swept.reduce((a, b) => {
      for (((bsA, seqA), (bsB, seqB)) <- a zip b) yield {
        (bsA + bsB, (seqA ++ seqB))
      }
    })
    val updates = for { ((startBS, collected), d) <- reduced.view.zipWithIndex } yield {
      val sortedLine = collected.sortBy(interval => interval._1)
      val intervals = ArrayBuffer[(Float, (Double, Double), BleuStats)]()
      sortedLine.foldLeft((intervals, startBS)) { (a, c) =>
        val (accum, bs) = a
        val (interval, diff) = c
        val currBS = bs + diff
        val entry = (interval, currBS.computeBleu(activationFactor), currBS)
        (accum += entry, currBS)
      }
      val sliding = intervals.sliding(2).toSeq
      val (update, (bleu, bp)) = if (sliding.size > 1) {
        val max = sliding.maxBy(interval => interval(0)._2._1)
        val best = max(1)
        val prev = max(0)
        //val (bleu, bp) = prev._2
        ((best._1 + prev._1) / 2, prev._2)
      } else {
        (Float.NaN, (0.0, 0.0))
      }
      if (!update.isNaN) {
        val updatedPoint = DenseVector(update, 1).t * Sweep.createProjectionMatrix(directions(d, ::).t, point)
        (updatedPoint, (bleu, bp))
      } else
        (point.t, (0.0, 0.0))
    }

    val serialisable = updates.toIndexedSeq

    val validated = for (vs <- validationSet) yield {
      val bsRDD = for ((mat, bleuStats) <- vs) yield {
        for ((param, (bleu, bp)) <- serialisable) yield {
          val scores = (param * mat).t
          val maxIndex = argmax(scores)
          bleuStats(maxIndex)
        }
      }
      bsRDD.reduce { (seqA, seqB) =>
        for ((a, b) <- seqA zip seqB) yield a + b
      }
    }
    val bestValidation = for (v <- validated) yield {
      val bleuScores = v.map { _.computeBleu() }
      val bestIndex = bleuScores.zipWithIndex.maxBy(_._1._1)._2
      (serialisable(bestIndex)._1, bleuScores(bestIndex))
    }
    bestValidation.getOrElse(updates.maxBy(_._2))
  }

  @tailrec
  def iterate(
    sc: SparkContext,
    prevPoint: DenseVector[Float],
    prevBleu: (Double, Double),
    nbests: RDD[BreezeNBest],
    deltaBleu: Double,
    r: RandBasis,
    noOfRandom: Int,
    affineDims: Set[Int],
    verify: Boolean,
    activationFactor: Option[Double],
    sweepFunc: Sweep.SweepFunc,
    validationSet: Option[RDD[BreezeNBest]]): (DenseVector[Float], (Double, Double)) = {
    val directions = generateDirections(r, prevPoint.length, noOfRandom, affineDims)
    val (point, (bleu, bp)) = iteration(nbests, prevPoint, directions, activationFactor, sweepFunc, validationSet)
    val (verifiedBleu, verifiedBP) = if (verify)
      computeScore(nbests, point.t).computeBleu(None)
    else
      (0.0, 0.0)
    println(point)
    val verifyString = if (verify) f" verified: $verifiedBleu%.6f [$verifiedBP%.4f]" else ""
    println(f"BLEU: $bleu%.6f [$bp%.4f]" + verifyString)
    if ((bleu - prevBleu._1) < deltaBleu)
      (prevPoint, prevBleu)
    else
      iterate(sc, point.t, (bleu, bp), nbests, deltaBleu, r, noOfRandom, affineDims, verify, activationFactor, sweepFunc, validationSet)
  }

  def doSmert(nbests: RDD[BreezeNBest], conf: Config, validationSet: Option[RDD[BreezeNBest]] = None)(implicit sc: SparkContext): (DenseVector[Float], (Double, Double)) = {
    import conf._
    val rb = new RandBasis(random)
    val initials = scala.collection.immutable.Vector(initialPoint) ++
      (for (i <- 0 until noOfInitials) yield {
        val tmp = DenseVector.rand(initialPoint.size, Gaussian(0, 1)(rb)).map(_.toFloat)
        affineDims.map { tmp(_) = 1f }
        tmp
      })
    val res = for (i <- initials)
      yield iterate(sc, i, (0.0, 0.0), nbests, deltaBleu, rb, noOfRandom, affineDims, verify, activationFactor, sweepFunc, validationSet)
    val (finalPoint, (finalBleu, finalBP)) = res.maxBy(_._2._1)
    println(f"Found final point with BLEU $finalBleu%.6f [$finalBP%.4f]!")
    (finalPoint, (finalBleu, finalBP))
  }

  def main(args: Array[String]): Unit = {
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
      opt[String]('b', "delta_bleu") action { (x, c) =>
        c.copy(deltaBleu = x.toDouble)
      } text ("Termination condition for the BLEU score")
      opt[Int]('r', "random_seed") action { (x, c) =>
        c.copy(random = getGenerator(x))
      } text ("Random Seed")
      opt[Int]('d', "directions") action { (x, c) =>
        c.copy(noOfRandom = x)
      } text ("Number of random directions")
      opt[Int]('q', "no_initials") action { (x, c) =>
        c.copy(noOfInitials = x)
      } text ("Number of randon initial points")
      opt[File]('o', "output") required () action { (x, c) =>
        c.copy(out = x)
      }
      opt[String]("objects") action { (x, c) =>
        c.copy(nbestsObjDir = Option(x))
      }
    }
    val cliConf = parser.parse(args, Config()).getOrElse(sys.exit())
    println(cliConf)
    import cliConf._
    val conf = new SparkConf().setAppName("SMERT").setIfMissing("spark.executor.heartbeatInterval", "100s")
    for (l <- localThreads) conf.setMaster(s"local[$l]")
    implicit val sc = new SparkContext(conf)
    sc.setCheckpointDir(".smertCheckpoint")
    val hypsRDD = nbestsObjDir.map(sc.objectFile[NBest](_)).getOrElse{
      val nbests = loadUCamNBest(nbestsDir)
      sc.parallelize(nbests).repartition(500)
    }
    val nbestsRDD = hypsRDD.map { n =>
        val mat = nbestToMatrix(n)
        val bs = n map (_.bs)
        (mat, bs)
    }.persist(StorageLevel.MEMORY_ONLY_2)
    nbestsRDD.checkpoint()
    val (finalPoint, _) = doSmert(nbestsRDD, cliConf)
    breeze.linalg.csvwrite(out, finalPoint.toDenseMatrix.map(_.toDouble))
    println(finalPoint)
  }
}
