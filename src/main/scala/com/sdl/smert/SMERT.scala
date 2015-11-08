package com.sdl.smert

import scala.collection.mutable.Buffer
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import com.sdl.NBest._
import java.io.File
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import com.sdl.BleuStats
import scala.annotation.tailrec
import scala.collection.mutable.ArrayBuffer

object SMERT {

  def doSvd(): Unit = {
    /*    val trainingSet : RDD[org.apache.spark.mllib.linalg.Vector] = for(nbest <- nbests; hyp <- nbest) yield{
      Vectors.dense(hyp.fVec.toArray)
    }
    trainin
    val trainingSetMat = new RowMatrix(trainingSet)
    val svd = trainingSetMat.computeSVD(, computeU = true)*/

  }
  
  def createProjectionMatrix(axis : Int, initial : Vector) = {
    val dir = DenseVector.zeros[F](initial.length)
    dir(axis) = 1
    DenseMatrix.vertcat(dir.toDenseMatrix, initial.toDenseMatrix)
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

  def main(args: Array[String]) {
    case class Config(nbestsDir: File = new File("."), initialPoint: Vector = DenseVector(), localThreads: Option[Int] = None, noOfPartitions : Int=100)
    val parser = new scopt.OptionParser[Config]("smert") {
      head("smert", "1.0")
      //NBest Option
      opt[File]('n', "nbest_dir") required () action { (x, c) =>
        c.copy(nbestsDir = x)
      } validate (x => if (x.isDirectory()) success else failure("Nbest directory is not a directory")) text ("NBest input directory")
      // Initial point
      opt[String]('i', "initial") required () action { (x, c) => 
        c.copy(initialPoint = stringToVec(x))   
      } text ("Initital starting point")
      opt[Int]('l', "local_threads") required () action { (x, c) => 
        c.copy(localThreads = Some(x))   
      } text ("If supplied, run in a local master with this number of threads")
      opt[Int]('p', "partitions") action { (x, c) => 
        c.copy(noOfPartitions = x)   
      } text ("Number of partitions")
    }
    val cliConf = parser.parse(args, Config()).getOrElse(sys.exit())
    println(cliConf)
    import cliConf._
    val conf = new SparkConf().setAppName("SMERT")
    for(l <- localThreads) conf.setMaster(s"local[$l]")
    val sc = new SparkContext(conf)
    val rawNbests = sc.parallelize(loadUCamNBest(nbestsDir), noOfPartitions)
    val nbests = (rawNbests map { n =>
      val mat: Matrix = n
      val bs = n map (_.bs)
      (mat, bs)
    }).cache
    
    
    val projection = createProjectionMatrix(0, initialPoint)
    
    val swept: RDD[Seq[(F, BleuStats, BleuStats)]] = for (nbest <- nbests) yield {
      val (mat, bs) = nbest
      val intervals = sweepLine(mat, projection)
      val diffs = for (((currInterval, currBS), i) <- intervals.view.zipWithIndex) yield {
        val prevBS = if (i == 0) currBS else intervals(i - 1)._2
        (currInterval, bs(currBS), bs(currBS) - bs(prevBS))
      }
      val res = diffs.toSeq
      res
    }
    val line = swept.flatMap(identity).sortBy(interval => interval._1, true)
    var error = BleuStats.empty
    val collected = for (intervalBoundary <- line.collect()) yield {
      val isInf = intervalBoundary._1 == Float.NegativeInfinity
      val bs = if (isInf) intervalBoundary._2 else intervalBoundary._3
      error = error + bs
      if (isInf)
        (intervalBoundary._1, bs.computeBleu(), bs)
      else
        (intervalBoundary._1, error.computeBleu(), error)
    }
    val max = collected.sliding(2).maxBy(interval => if (interval(0)._1 == F.NegativeInfinity) F.MinValue else interval(1)._2._1)
    val best = max(1)
    val (bleu, bp) = best._2
    val prev = max(0)
    val update = (best._1 - prev._1)/2
    println(f"Update at $update%s with BLEU: $bleu%.3f [$bp%.4f]")
    val updatedPoint = DenseVector(update, 1).t * projection 
    println(updatedPoint)
  }

}