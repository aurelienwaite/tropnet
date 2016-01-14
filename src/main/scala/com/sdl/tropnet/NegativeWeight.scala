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
import org.apache.commons.math3.random.RandomGenerator

object NegativeWeight {

  /**
   * Create the fire vectors.
   */
  def fireVectors(in: DenseMatrix[Float], params: Seq[DenseVector[Float]], bleuStats: IndexedSeq[BleuStats]) = {
    val fireVecs = ArrayBuffer[DenseVector[Float]]()
    val fireBS = ArrayBuffer[BleuStats]()
    val ones = DenseMatrix.ones[Float](1, in.cols)
    val withBias = DenseMatrix.vertcat(ones, in)
    val activations = (for (p <- params) yield (p.t * withBias).t.map(math.max(_, 0))).reduce(_ + _)
    val projected = DenseMatrix.vertcat(withBias, activations.toDenseMatrix)
    for (c <- 0 until projected.cols) {
      val initial = projected(-1, c)
      if (initial != 0f) {
        val fireVec = DenseVector.zeros[Float](projected.rows)
        fireVec(-1) = initial
        fireVecs += fireVec
        fireBS += bleuStats(c)
      }
      val row = projected(::, c)
      row(-1) = initial
      fireVecs += row
      fireBS += bleuStats(c)
    }
    fireVecs += DenseVector.zeros[Float](projected.rows)
    fireBS += BleuStats.bad
    //println(s"${projected.cols} ${fireVecs.size}")
    (DenseMatrix.horzcat(fireVecs.map(_.toDenseMatrix.t): _*), fireBS)
  }

  /*def expandMatrix(in : DenseMatrix[Float], size: Int) = {
    val withBias =  DenseMatrix.vertcat(DenseMatrix.ones[Float](1, in.cols), in)
    DenseMatrix.vertcat(Seq.fill(size)(withBias):_*)
  }
  
  def createProjection(params : Seq[DenseVector[Float]]) = {
    val eye = DenseMatrix.eye[Float](13) 
    val zeros = DenseMatrix.zeros[Float](13, 13 * params.size)
    val dirs = DenseMatrix.horzcat(zeros, eye) 
    val initials = DenseMatrix.horzcat(
          params.map(_.toDenseMatrix) :+ 
          DenseMatrix.zeros[Float](1, 13) :_*)
    DenseMatrix.vertcat(dirs,initials)     
  }*/

  def printNeurons(neurons: Seq[DenseVector[Float]]) = {
    for ((neuron, i) <- neurons.view.zipWithIndex) {
      println(s"Neuron $i  " + neuron.toArray.map(formatter.format(_)).mkString(","))
    }
  }

  def optimiseNeuron(nbests: Seq[NBest], paramVec: DenseVector[Float],
                     other: Seq[DenseVector[Float]], r: RandomGenerator)(implicit sc: SparkContext) = {
    val input = for {
      nbest <- nbests
      bs = nbest map (_.bs)
      mat = SMERT.nbestToMatrix(nbest)
      (fire, fireBS) = fireVectors(mat, other, bs)
    } yield (fire, fireBS)
    val smertInitial = DenseVector.vertcat(paramVec, DenseVector.ones[Float](1))
    val conf = SMERT.Config(
      initialPoint = smertInitial,
      affineDim = Some(13),
      noOfInitials = 5,
      noOfRandom = 28,
      random = r)
    val (point, (newBleu, bp)) = SMERT.doSmert(input.seq, conf)
    val res = point(0 to -2) +: other
    (res, (newBleu, bp))
  }

  def isolateNeurons[T](prev: List[T], next: List[T]): List[(T, List[T])] =
    next match {
      case Nil          => Nil
      case head :: tail => (head, prev ++ tail) +: isolateNeurons(prev :+ head, tail)
    }

  @tailrec
  def iterate(nbests: Seq[NBest], params: List[DenseVector[Float]], bleu: Double, r: RandomGenerator)(implicit sc: SparkContext): Seq[DenseVector[Float]] = {
    val isolated = isolateNeurons(Nil, params)
    val optimised = for ((p, other) <- isolated.drop(1)) yield {
      printNeurons(p +: other)
      optimiseNeuron(nbests, p, other, r)
    }
    val (res, (newBleu, bp)) = optimised.maxBy(_._2._1)
    //if (newBleu - bleu < SMERT.Config().deltaBleu)
    //  params
    //else {
    println(s"Caetano iteration done: SnewBleu")
    printNeurons(res)
    iterate(nbests, res.toList, newBleu, r)
    //}
  }

  def main(args: Array[String]): Unit = {

    val NO_OF_UNITS = args(1).toInt

    val sparkConf = new SparkConf().setAppName("Negative Weight")
    sparkConf.setMaster("local[8]")
    implicit val sc = new SparkContext(sparkConf)

    val nbests = loadUCamNBest(new File(args(0)))
    val MAGIC_BIAS = 250.0

    val initialisedUnit = DenseVector(0, 1.000000, 0.820073, 1.048347, 0.798443, 0.349793, 0.286489, 15.352371, -5.753633, -3.766533, 0.052922, 0.624889, -0.015877).map(_.toFloat)
    val magicUnit = DenseVector(MAGIC_BIAS, 1.000000, 0.820073, 1.048347, 0.798443, 0.349793, 0.286489, 15.352371, -5.753633, -3.766533, 0.052922, 0.624889, -0.015877).map(_.toFloat)

    val flat = DenseVector.ones[Float](13)
    val neurons = List.fill(NO_OF_UNITS)(initialisedUnit)

    val nn = iterate(nbests, neurons, 0, SMERT.getGenerator(11))
    printNeurons(nn)
  }

}
