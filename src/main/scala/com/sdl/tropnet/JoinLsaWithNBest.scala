package com.sdl.tropnet

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import com.sdl.NBest._
import java.io.File
import com.sdl.smert.SMERT
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.Vectors
import breeze.stats.distributions.RandBasis
import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.distributed.IndexedRow
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix
import java.io.ObjectOutputStream
import java.io.FileOutputStream
import org.apache.hadoop.io.NullWritable
import org.apache.hadoop.io.BytesWritable
import org.apache.spark.util.Utils
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.Row

/**
 * Do SVD on an NBest list. Command line arguments
 * 
 * 0 - The NBest list directory
 * 1 - A location on HDFS for the U matrix
 * 2 - 
 */
object JoinLsaWithNBest {
  def main(args: Array[String]): Unit = {

    val sparkConf = new SparkConf().setAppName("TropNet")
    implicit val sc = new SparkContext(sparkConf)
    //sc.setCheckpointDir(".tropnetCheckpoint")

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val lsaU = sqlContext.read.parquet(args(1)).rdd.map { 
      case Row(index: Long, features: Vector) => (index, features)
    }
    
    val nbests = loadUCamNBest(new File(args(0)))
    val flattened = for { 
      (nbest, sentIndex) <- nbests.zipWithIndex
      (hyp, hypIndex) <- nbest.zipWithIndex
    } yield (hyp, sentIndex, hypIndex)
    val flattendRDD = sc.parallelize(flattened, 500).zipWithIndex().map{ case (k,v) => (v, k)}
    
    val joined = flattendRDD join lsaU
    
    val appendedFeatureVecs = joined.map { case (_, ((hyp, sentIndex, hypIndex), lsaFVec)) => 
      val converted = lsaFVec.toArray.map(_.toDouble)
      val newHyp = hyp.copy(fVec = hyp.fVec ++ converted)
      (newHyp, sentIndex, hypIndex)
    }
    val grouped = appendedFeatureVecs.groupBy(_._2)
    val appendedNbests = grouped.map{ case (sentIndex, nbestIt) => 
      val nbest: NBest = nbestIt.toIndexedSeq.sortBy(_._3).map(_._1)
      (sentIndex, nbest)
    }.sortBy(_._1).map(_._2).cache
    appendedNbests.saveAsObjectFile(args(2))
    for(n <- appendedNbests.take(2)) println(n)
  }

}