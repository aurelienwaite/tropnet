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
import scala.annotation.tailrec

object NegativeWeight {

  /**
   * Create the fire vectors. 
   */
  def fireVectors(projected: DenseMatrix[Float], biases: (Float, Float), bleuStats : IndexedSeq[BleuStats] ) = {
    val fireVecs = ArrayBuffer[DenseVector[Float]]()
    val fireBS = ArrayBuffer[BleuStats]()
    for(c <- 0 until projected.cols) {
      val initial = math.max(projected(-1, c),0f) * biases._1
      if (initial != 0f){
        val fireVec = DenseVector.zeros[Float](projected.rows)
        fireVec(-1) = initial
        fireVecs += fireVec
        fireBS += bleuStats(c)
      }
      val row = projected(::, c).map(_*  biases._2)
      row(-1) = initial
      fireVecs += row
      fireBS += bleuStats(c)
    }
    fireVecs += DenseVector.zeros[Float](projected.rows)
    fireBS += bleuStats(0)
    println(s"${projected.cols} ${fireVecs.size}")
    (DenseMatrix.horzcat(fireVecs.map(_.toDenseMatrix.t):_*), fireBS)
  }
  
  def expandMatrix(in : DenseMatrix[Float]) = DenseMatrix.vertcat(DenseMatrix.ones[Float](1, in.cols), in, DenseMatrix.ones[Float](1, in.cols), in )
  
  def createProjection(params : DenseVector[Float]) = {
    val eye = DenseMatrix.eye[Float](13)
    val zeros = DenseMatrix.zeros[Float](13, 13)
    val dirs = DenseMatrix.horzcat(zeros, eye) 
    val initials = DenseMatrix.horzcat(
          params.toDenseMatrix,
          DenseMatrix.zeros[Float](1, 13))
    DenseMatrix.vertcat(dirs,initials)     
  }
  
  @tailrec
  def iterate(nbests: Seq[NBest], params: (DenseVector[Float], DenseVector[Float]), bleu: Double)(implicit sc: SparkContext) 
    : (DenseVector[Float], DenseVector[Float]) = {
    val (p1, p2) = params
    val projection = createProjection(p1)
    val input = for {
      nbest <- nbests
      bs = nbest map (_.bs)
      mat = SMERT.nbestToMatrix(nbest)
      expanded = expandMatrix(mat) 
      projected = projection * expanded
      (fire, fireBS) = fireVectors(projected, (1.0f, 1.0f), bs)
    } yield (fire, fireBS)
    
    val smertInitial = DenseVector.vertcat(p2, DenseVector.ones[Float](1))
    val conf = SMERT.Config(initialPoint = smertInitial, affineDim = Some(13))
    val (point, (newBleu,bp)) = SMERT.doSmert(input.seq, conf)
    val res = (point(0 to -2), p1)
    if (newBleu < bleu) 
      res
    else
      iterate(nbests, res, newBleu)
  }
  
  def main(args: Array[String]): Unit = {
   
    val sparkConf = new SparkConf().setAppName("Negative Weight")
    sparkConf.setMaster("local[8]")
    implicit val sc = new SparkContext(sparkConf)

   val nbests = loadUCamNBest(new File(args(0)))
       val MAGIC_BIAS = 250.0
   val firstUnit = DenseVector(MAGIC_BIAS,1.000000,0.820073,1.048347,0.798443,0.349793,0.286489,15.352371,-5.753633,-3.766533,0.052922,0.624889,-0.015877).map(_.toFloat) 
   val secondUnit = DenseVector(0,1.000000,0.820073,1.048347,0.798443,0.349793,0.286489,15.352371,-5.753633,-3.766533,0.052922,0.624889,-0.015877).map(_.toFloat)
   val nn = iterate(nbests, (firstUnit, secondUnit), 0)
   println(nn)
  }

}
