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
import breeze.stats.distributions.Gaussian
import breeze.stats.distributions.RandBasis
import com.sdl.smert.Sweep._
import com.sdl.smert.MaxiMinSweep

object Caetano {

  case class Neuron(params: DenseVector[Float], multiplier: Float){
    override def toString() = {
      val p = params.toArray.map(formatter.format(_)).mkString(",")
      val multiplierString = formatter.format(multiplier)
      s"multiplier:$multiplier;params:$p"
    }
  }
  
  def withBiases(mat: DenseMatrix[Float], others: Seq[Neuron]) = {
    val ones = DenseMatrix.ones[Float](1, mat.cols)
    val withBias = DenseMatrix.vertcat(ones, mat)
    val activations = (for (n <- others) yield (n.params.t * withBias).t.map(math.max(_, 0)))
    val activationsMatrix = DenseMatrix.vertcat(activations.map(_.toDenseMatrix): _*)
    val res = DenseMatrix.vertcat(withBias, activationsMatrix)
    res
  }
  
  /**
   * Create the fire vectors.
   */
  def fireVectors(in: DenseMatrix[Float], neurons: Seq[Neuron], bleuStats: IndexedSeq[BleuStats]) = {
    val fireVecs = ArrayBuffer[DenseVector[Float]]()
    val fireBS = ArrayBuffer[BleuStats]()
    val ones = DenseMatrix.ones[Float](1, in.cols)
    //val absolute = math.abs(multiplier)
    val withBias = DenseMatrix.vertcat(ones, in)
    //val scaled = withBias.map(_ / multiplier)
    val activations = (for (n <- neurons) yield (n.params.t * withBias).t.map(math.max(_, 0)))
    val activationSum = activations.reduce(_+_)
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
    fireVecs += DenseVector.zeros[Float](withBias.rows + neurons.size)
    fireBS += BleuStats.bad
    val out = DenseMatrix.horzcat(fireVecs.map(_.toDenseMatrix.t): _*)
    //val res = if(multiplier < 0) out.map(_ * -1) else out
    (out, fireBS)
  }
  
  


  def printNeurons(neurons: Seq[Neuron]) = 
    for ((neuron, i) <- neurons.view.zipWithIndex) 
      println(s"Neuron $i  $neuron")
    
  

  def optimiseNeuron(nbests: Seq[NBest], toOptimise: Neuron,
                     other: Seq[Neuron], r: RandomGenerator)(implicit sc: SparkContext) = {
    val input = for {
      nbest <- nbests
      bs = nbest map (_.bs)
      mat = SMERT.nbestToMatrix(nbest)
      //(fire, fireBS) = fireVectors(mat, other, bs)
    } yield if (toOptimise.multiplier < 0) 
      (withBiases(mat, other), bs)
    else
      fireVectors(mat, other, bs)
    //breeze.linalg.csvwrite(new File("/tmp/fire"), input(0)._1.map(_.toDouble))
    val smertInitial = DenseVector.vertcat(toOptimise.params :* toOptimise.multiplier, DenseVector(other.map(_.multiplier.toFloat) :_*))
    val(noOfInitials, noOfRandom, sweepFunc: SweepFunc) = if(toOptimise.multiplier < 0) (
       0, 
       0, 
       MaxiMinSweep.maxiMinSweepLine(0 until (toOptimise.params.length))_
      )
    else (
      10,
      39,
      sweepLine _
    )
    val conf = SMERT.Config(
      initialPoint = smertInitial,
      noOfInitials = noOfInitials,
      noOfRandom = noOfRandom,
      sweepFunc = sweepFunc,
      random = r,
      activationFactor = Option(0.01))
    val (point, (newBleu, bp)) = SMERT.doSmert(input.seq, conf)
    val newMultiplier = if (toOptimise.multiplier < 0) -1.0f else 1.0f
    val newPoint = point(0 until toOptimise.params.length) :* newMultiplier
    val multipliers = point(toOptimise.params.length to -1).toArray
    val withMultipliers = for ((n, m) <- other zip multipliers) yield Neuron(n.params, m)
    val res = Neuron(newPoint, newMultiplier) +: withMultipliers
    (res , (newBleu, bp))
  }

  def isolateNeurons[T](prev: List[T], next: List[T]): List[(T, List[T])] =
    next match {
      case Nil          => Nil
      case head :: tail => (head, prev ++ tail) +: isolateNeurons(prev :+ head, tail)
    }

  @tailrec
  def iterate(nbests: Seq[NBest], neurons: List[Neuron], bleu: Double, r: RandomGenerator)(implicit sc: SparkContext): Seq[Neuron] = {
    val isolated = isolateNeurons(Nil, neurons)
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

    val sparkConf = new SparkConf().setAppName("Caetano")
    sparkConf.setMaster("local[8]")
    implicit val sc = new SparkContext(sparkConf)

    val nbests = loadUCamNBest(new File(args(0)))

    val generator = SMERT.getGenerator(11)
    val rb = new RandBasis(generator)
    
    //val initialisedParams = DenseVector(1.0,2.4906442165374756,1.854056477546692,2.6558055877685547,3.3370068073272705,-2.2473344802856445,1.281643271446228,41.036888122558594,-16.5937442779541,-12.700494766235352,-0.45229414105415344,2.266307830810547,0.14411388337612152).map(_.toFloat)
    //val flat = DenseVector.ones[Float](13)
    
    val neurons = List.fill(NO_OF_UNITS){
      val n = Neuron(DenseVector.rand(13, Gaussian(0, 1)(rb).map(_.toFloat)), 1)
      //hack to ensure all units fire at start
      n.params(0) = 999f
      n
    }
    

    val nn = iterate(nbests, neurons, 0, generator)
    printNeurons(nn)
  }

}
