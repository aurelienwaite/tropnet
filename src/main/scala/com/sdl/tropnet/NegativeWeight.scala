package com.sdl.tropnet

import java.io.FileWriter
import scala.util.Random
import resource._
import com.sdl.NBest._
import com.sdl.tropnet.CreateFireVectors._
import java.io.File
import com.sdl.smert.SMERT
import breeze.linalg._
import breeze.numerics._
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import scala.collection.mutable.ArrayBuffer
import com.sdl.BleuStats

object NegativeWeight {

  /**
   * Create the fire vectors. 
   */
  def fireVectors(projected: DenseMatrix[Float], biases: (Float, Float), bleuStats : IndexedSeq[BleuStats] ) = {
    val fireVecs = ArrayBuffer[DenseVector[Float]]()
    val fireBS = ArrayBuffer[BleuStats]()
    for(r <- 0 until projected.rows) {
      val initial = math.max(projected(r, -1),0f) * biases._1
      if (initial != 0f){
        val fireVec = DenseVector.zeros[Float](projected.cols)
        fireVec(-1) = initial
        fireVecs += fireVec
        fireBS += bleuStats(r)
      }
      val row = projected(r, ::).t.map(_*  biases._2)
      row(-1) = initial
      fireVecs += row
      fireBS += bleuStats(r)
    }
    fireVecs += DenseVector.zeros[Float](projected.cols)
    fireBS += BleuStats.bad
    (DenseMatrix.vertcat(fireVecs.map(_.toDenseMatrix):_*), fireBS)
  }
  
  def expandMatrix(in : DenseMatrix[Float]) = DenseMatrix.horzcat(DenseMatrix.ones[Float](in.rows, 1), in, DenseMatrix.ones[Float](in.rows, 1), in )
  
  def main(args: Array[String]): Unit = {
    val i= DenseMatrix.eye[Float](13)
    val zeros = DenseMatrix.zeros[Float](13, 13)
    //val biasDir = DenseMatrix.zeros[Float](1, 26)
    //biasDir(0,0) = 1.0f
    val dirs = DenseMatrix.vertcat(DenseMatrix.horzcat(zeros, i))
    val initial = DenseMatrix.horzcat(
        DenseMatrix(25.0, 1.000000,0.820073,1.048347,0.798443,0.349793,0.286489,15.352371,-5.753633,-3.766533,0.052922,0.624889,-0.015877).t,
        DenseMatrix.zeros[Double](1, 13)
        ).map(_.toFloat)
    val projection = DenseMatrix.vertcat(dirs, initial)  
    
    val nbests = loadUCamNBest(new File(args(0))).par
    val input = for {
      nbest <- nbests
      bs = nbest map (_.bs)
      mat = SMERT.nbestToMatrix(nbest)
      projected = projection * mat 
      (fire, fireBS) = fireVectors(expandMatrix(projected), (1.0f, -0.25f), bs)
      negated = fire.map(_ * -1.0f)
    } yield (negated, fireBS)
    
    sys.exit()
    
    /*val params = DenseVector(4.444362, 6.521777, 10.26614, 11.057872, -4.727493, 18.885849, 12.249861, -14.08159, -14.655475, 3.0803752, 0.6224642, -10.696804, -1.7164031, -3.6025164, -1.36588).map(_.toFloat).t
    val bleu = input.seq.map { case (proj, bs) =>
      val scores = (params * proj).t
      bs(argsort(scores).last)
    }.reduce(_+_)
    println(bleu.computeBleu())
    sys.exit*/
    
    val sparkConf = new SparkConf().setAppName("Negative Weight")
    sparkConf.setMaster("local[8]")
    implicit val sc = new SparkContext(sparkConf)
    val smertInitial = DenseVector(0,0,0,0,0,0,0,0,0,0,0,0,0,0,1).map(_.toFloat)
    val conf = SMERT.Config(initialPoint = smertInitial, affineDim = Some(13))
    SMERT.doSmert(input.seq, conf)  
  }

}
