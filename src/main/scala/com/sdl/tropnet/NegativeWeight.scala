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
    val activations = (for (p <- params) yield (p.t * withBias).t.map(math.max(_, 0)))
    val activationSum = activations.reduce(_+_)
    //val projected = DenseMatrix.vertcat((withBias +: activations.map(_.toDenseMatrix) ) :_* )
    for (c <- 0 until withBias.cols) {
      val activationsVector = DenseVector.vertcat(activations.map(a=>a(c to c)):_*)
      if (activationSum(c) != 0f) {
        val fireVec = DenseVector.vertcat(DenseVector.zeros[Float](withBias.rows), activationsVector)
        fireVecs += fireVec
        fireBS += bleuStats(c).deleteActivations()
      }
      val row = DenseVector.vertcat(withBias(::, c), activationsVector)
      fireVecs += row
      fireBS += bleuStats(c)
    }
    fireVecs += DenseVector.zeros[Float](withBias.rows + params.size)
    fireBS += BleuStats.bad
    (DenseMatrix.horzcat(fireVecs.map(_.toDenseMatrix.t): _*), fireBS)
  }


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
    //breeze.linalg.csvwrite(new File("/tmp/fire"), input(0)._1.map(_.toDouble))
    val smertInitial = DenseVector.vertcat(paramVec, DenseVector.ones[Float](other.size))
    val conf = SMERT.Config(
      initialPoint = smertInitial,
      affineDims = (paramVec.size until (paramVec.size + other.length)).toSet,
      noOfInitials = 10,
      noOfRandom = 39,
      random = r,
      activationFactor = Option(0.01))
    val (point, (newBleu, bp)) = SMERT.doSmert(input.seq, conf)
    val res = point(0 until paramVec.length) +: other
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
    println(s"Caetano iteration done: $newBleu")
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

    val initialisedUnit = DenseVector(1.0,2.4906442165374756,1.854056477546692,2.6558055877685547,3.3370068073272705,-2.2473344802856445,1.281643271446228,41.036888122558594,-16.5937442779541,-12.700494766235352,-0.45229414105415344,2.266307830810547,0.14411388337612152).map(_.toFloat)

    val flat = DenseVector.ones[Float](13)
    val neurons = List.fill(NO_OF_UNITS)(initialisedUnit)

    val nn = iterate(nbests, neurons, 0, SMERT.getGenerator(11))
    printNeurons(nn)
  }

}
