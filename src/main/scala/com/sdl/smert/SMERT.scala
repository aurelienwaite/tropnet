package com.sdl.smert

import scala.collection.mutable.Buffer
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import com.sdl.NBest._
import java.io.File
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.Vectors
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

  @tailrec
  def applyError(prevErr: BleuStats, intervals: Seq[(Float, _, BleuStats)], accum: Buffer[(Float, (Double, Double), BleuStats)]): Unit = {
    intervals match {
      case Nil => Unit
      case head +: tail => {
        val err = if(head._1 != Float.NegativeInfinity) prevErr + head._3 else prevErr
        val res = (head._1, err.computeBleu(), err)
        accum += res
        applyError(err, tail, accum)
      }
    }
  }

  def main(args: Array[String]) {
    
    val noOfPartitions=50
    
    val conf = new SparkConf().setAppName("SMERT").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val rawNbests = sc.parallelize(loadUCamNBest(new File(args(0))), noOfPartitions)
    val nbests = (rawNbests map { n =>
      val mat: breeze.linalg.DenseMatrix[Float] = n
      val bs = n map (_.bs)
      (mat, bs)
    }).cache
    val projection = breeze.linalg.DenseMatrix(
      (1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f),
      (1.000000f, 0.820073f, 1.048347f, 0.798443f, 0.349793f, 0.286489f, 15.352371f, -5.753633f, -3.766533f, 0.052922f, 0.624889f, -0.015877f))

    //Start of distributed sweep line
    val swept: RDD[Seq[(Float, BleuStats, BleuStats)]] = for (nbest <- nbests) yield {
      val (mat, bs) = nbest
      val intervals = sweepLine(mat, projection)
      val diffs = for (((currInterval, currBS),i) <- intervals.view.zipWithIndex) yield {
        val prevBS = if (i ==0) currBS else intervals(i-1)._2
        (currInterval, bs(currBS), bs(currBS) - bs(prevBS))
      }
      val res = diffs.toSeq
      //println(res.toIndexedSeq)
      res
    }
    val line = swept.flatMap(identity).sortBy(interval => interval._1, true)
    var error = BleuStats.empty
    val collected = for(intervalBoundary <- line.collect()) yield {
      val isInf = intervalBoundary._1 == Float.NegativeInfinity
      val bs = if (isInf)intervalBoundary._2 else intervalBoundary._3 
      error =  error + bs
      if (isInf) 
        (intervalBoundary._1, bs.computeBleu(), bs)
      else
        (intervalBoundary._1, error.computeBleu(), error)
    }
    /*
    val partitions = line.mapPartitionsWithIndex((i, partition) => Seq((i, partition.toIndexedSeq)).toIterator, true)
    val startIntervals = partitions.flatMap { p =>
      (for (h <- p._2.headOption) yield (Seq((p._1, h._1)))).getOrElse(Seq.empty)
    }.coalesce(1)
    var c = startIntervals.cartesian(swept)
    val startError = for (((i, sI), nbestLine) <- c) yield {
      assert(!nbestLine.isEmpty, "Empty NBest list")
      val bs = if (sI == Float.NegativeInfinity || sI <= nbestLine.head._1)
        nbestLine.head._2
      else if (sI > nbestLine.last._1)
        nbestLine.last._2
      else {
        val interval = nbestLine.sliding(2) find { interval =>
          {
            val start = interval(0)
            val end = interval(1)
            sI > start._1 && sI <= end._1
          }
        }
        interval.map(_(1)._2).getOrElse(sys.error(s"No interval found for start $sI and line $nbestLine"))
      }
      (i, bs)
    }
    val aggregated = startError.reduceByKey(_ + _)
    val errorSurface = partitions.join(aggregated) flatMap {
      case (i, (partition, startError)) => {
        val res = ArrayBuffer[(Float, (Double, Double), BleuStats)]()
        res.sizeHint(partition.size)
        applyError(startError, partition, res)
        res
      }
    }*/

    //val collected = errorSurface.collect().sortWith(_._1 < _._1)
    for ((intervalBoundary, (bleu, bp), bs) <- collected) {
      println(s"$intervalBoundary\t$bleu [$bp]\t$bs")
    }   
    
    //errorSurface.foreach(println(_))

  }

}