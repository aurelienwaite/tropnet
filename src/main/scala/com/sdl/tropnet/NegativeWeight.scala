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

object NegativeWeight {

  def main(args: Array[String]): Unit = {
    val i= DenseMatrix.eye[Float](13)
    val zeros = DenseMatrix.zeros[Float](13, 13)
    val biasDir = DenseMatrix.zeros[Float](1, 26)
    biasDir(0,0) = 1.0f
    val dirs = DenseMatrix.vertcat(biasDir,DenseMatrix.horzcat(zeros, i))
    val initial = DenseMatrix.horzcat(
        DenseMatrix(0.0, 1.000000,0.820073,1.048347,0.798443,0.349793,0.286489,15.352371,-5.753633,-3.766533,0.052922,0.624889,-0.015877).t,
        DenseMatrix.zeros[Double](1, 13)
        ).map(_.toFloat)
    val projection = DenseMatrix.vertcat(dirs, initial)  
    
    val nbests = loadUCamNBest(new File(args(0))).par
    val input = for {
      nbest <- nbests
      fire = createFireVectors(nbest, (1.0, -0.25))
      mat = SMERT.nbestToMatrix(fire)
      projected = projection * mat  
      negated = projected.map(_ * -1.0f)
      bs = nbest map (_.bs)
    } yield (projected, bs)
    val sparkConf = new SparkConf().setAppName("Negative Weight")
    sparkConf.setMaster("local[8]")
    implicit val sc = new SparkContext(sparkConf)
    val smertInitial = DenseVector(0,0,0,0,0,0,0,0,0,0,0,0,0,0,1).map(_.toFloat)
    val conf = SMERT.Config(initialPoint = smertInitial)
    SMERT.doSmert(input.seq, conf)  
  }

}
