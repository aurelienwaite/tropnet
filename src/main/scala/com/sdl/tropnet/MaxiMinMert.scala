package com.sdl.tropnet

import com.sdl.tropnet.Caetano.Neuron
import breeze.linalg.DenseVector
import java.io.File
import com.sdl.smert.SMERT
import breeze.linalg.DenseMatrix
import breeze.stats.distributions.Gaussian
import org.apache.commons.math3.random.MersenneTwister
import breeze.stats.distributions.RandBasis
import com.sdl.smert.L
import com.sdl.BleuStats
import com.sdl.smert.Sweep
import scala.collection.mutable.Buffer
import scala.annotation.tailrec
import breeze.linalg.sum
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import com.sdl.smert.MaxiMinSweep

object MaxiMinMert extends App {

  val toOptimise = Neuron(DenseVector(999, -0.513182, 0.07114, -0.260172, -0.719798, 1.876827, -0.243718, 0.693919, -0.85612, 0.368958, -1.216058, -0.356035, 0.689836).map(_.toFloat), -0.50416815f)
  val others = Seq(
    Neuron(DenseVector(1001.840088, -0.718708, 3.389812, -0.141771, -0.158513, -4.388076, 3.203126, 32.554333, -13.840285, -4.114048, 1.954047, 1.470156, -1.594916).map(_.toFloat), 1f),
    Neuron(DenseVector(999, 1.615786, -1.373038, 1.560508, 1.37906, -0.800385, -1.347738, -0.275455, 0.239759, -1.360193, 1.216088, -0.145075, 1.393733).map(_.toFloat), 0.689836f))

  import com.sdl.NBest._
  val nbests = loadUCamNBest(new File(args(0)))

  val sparkConf = new SparkConf().setAppName("MaxiMin")
  sparkConf.setMaster("local[1]")
  implicit val sc = new SparkContext(sparkConf)

  val input = for {
    nbest <- nbests
    bs = nbest map (_.bs)
    mat = SMERT.nbestToMatrix(nbest)
    ones = DenseMatrix.ones[Float](1, mat.cols)
    withBias = DenseMatrix.vertcat(ones, mat)
    scaled = withBias.map(_ / toOptimise.multiplier)
  } yield {
    val activations = (for (n <- others) yield (n.params.t * withBias).t.map(math.max(_, 0)))
    val activationsMatrix = DenseMatrix.vertcat(activations.map(_.toDenseMatrix): _*)
    val res = DenseMatrix.vertcat(withBias, activationsMatrix)
    (res, bs)
  }
  breeze.linalg.csvwrite(new File("/tmp/fire"), input(0)._1.map(_.toDouble))
  val smertInitial = DenseVector.vertcat(toOptimise.params, DenseVector(others.map(_.multiplier.toFloat): _*))
  val conf = SMERT.Config(
    initialPoint = smertInitial,
    noOfInitials = 0,
    sweepFunc = MaxiMinSweep.maxiMinSweepLine(0 until (toOptimise.params.length + 1))_,
    noOfRandom = 0 //,
    //activationFactor = Option(0.01)
    )
  val (point, (newBleu, bp)) = SMERT.doSmert(input.seq, conf)
  val newPoint = point(0 until toOptimise.params.length)
  val multipliers = point(toOptimise.params.length to -1).toArray
  val withMultipliers = for ((n, m) <- others zip multipliers) yield Neuron(n.params, m)
  val res = Neuron(newPoint, toOptimise.multiplier) +: withMultipliers
  (res, (newBleu, bp))

  /*val r = new RandBasis(new MersenneTwister(11l))
  val randDir = DenseMatrix.rand(1, toOptimise.params.size + others.size, Gaussian(0, 1)(r).map(_.toFloat))
  val initial = DenseVector ((toOptimise.params.toScalaVector() ++ others.map(_.multiplier)) :_*).toDenseMatrix
  val projection = DenseMatrix.vertcat(randDir, initial)
  
  
  
  val input = for {
    nbest <- nbests
    bs = nbest map (_.bs)
    mat = SMERT.nbestToMatrix(nbest)
  } yield {
    val ones = DenseMatrix.ones[Float](1, mat.cols)
    val withBias = DenseMatrix.vertcat(ones, mat)
    val scaled = withBias.map(_ / toOptimise.multiplier)
    val activations = for (n <- others) yield (n.params.t * withBias).t.map(math.max(0,_)).toDenseMatrix
    val activationsMatrix = DenseMatrix.vertcat(activations:_*)
    val activated = DenseMatrix.vertcat(scaled, activationsMatrix)
    val deactivated = DenseMatrix.vertcat(DenseMatrix.zeros[Float](scaled.rows, scaled.cols), activationsMatrix)
    (activated, deactivated)
  }

  val (activated, deactivated) = input.head
  val bs = nbests.head.map(_.bs)
  //Matrix looks good
  //breeze.linalg.csvwrite(new File("/tmp/fire"), forTesting.map(_.toDouble))
  
  val projectedActivated = projection * activated
  val projectedDeactivated = projection * deactivated
  
  breeze.linalg.csvwrite(new File("/tmp/projectedActivated"), projectedActivated.map(_.toDouble))
  breeze.linalg.csvwrite(new File("/tmp/projectedDeactivated"), projectedDeactivated.map(_.toDouble))
  
  val minksum = for (c <- 0 until activated.cols) yield {
    val deactivatedBS = if (sum(projectedDeactivated(::, c)) == 0.0) BleuStats.bad else bs(c).deleteActivations()
    val l = Seq((L(projectedActivated(::, c),c), bs(c)) , (L(projectedDeactivated(::, c),c), deactivatedBS)).sortBy(_._1.m)
    val left = l(0)._1
    val right = l(1)._1
    // Note that the second line is placed first, because we want the min
    left.x = (right.y - left.y) / (left.m - right.m)
    (l(1), l(0), c)
  }
  
  val sorted = minksum.sortBy(_._2._1.x)
  val initials = for(s<-sorted) yield (s._1)
  
  def update(slice: DenseVector[Float], l: L) ={
    slice(0) = l.m
    slice(1) = l.y
  }
  
  def nbest2Mat(nbest : IndexedSeq[(L, BleuStats)]) = {
    val nbestMat = DenseMatrix.zeros[Float](2, nbest.size)
    for(((l,_), i) <- nbest.view.zipWithIndex) update(nbestMat(::, i), l)
    (nbestMat, nbest.map(_._2))
  }
  
  val minned = sorted.foldLeft(List((nbest2Mat(initials), Float.NegativeInfinity))){ (prev, s) =>
    val (_, (l,bs), index) = s
    val (nbest, _) = prev.last
    val updated = nbest._1.copy
    update(updated(::, index), l)
    val bsUpdated = nbest._2.updated(index, bs)
    prev :+ ((updated, bsUpdated), l.x)    
  }

  @tailrec
  def sweepOverMinned(in: List[((DenseMatrix[Float], IndexedSeq[BleuStats]), Float)], accum: Buffer[(Float, BleuStats)]) 
    : Seq[(Float, Int)]= {
    in match {
      case Nil => Nil
      case head :: tail => {
        val ((mat, bs), interval) = head
        val swept = Sweep.sweepLine(mat)
        val nextInterval = tail match {
          case Nil => Float.PositiveInfinity
          case (_,i) :: _ => i
        }
        @tailrec
        def filterSwept(toFilter: List[(Float, Int)], accum: Buffer[(Float, Int)]) : Unit = toFilter match {
          case Nil => Unit
          case head :: tail => {
            val (sweepInterval, _) = head
            val sweepNextInterval = tail match {
              case Nil => Float.PositiveInfinity
              case (e, _) :: _ => e 
            }
            if(
                (sweepInterval >= interval && sweepInterval < nextInterval) ||
                (sweepInterval <= interval &&  sweepNextInterval > nextInterval)
              ) accum += head
            filterSwept(tail, accum)
          }
        }
        val filtered =  Buffer[(Float, Int)]()
        filterSwept(swept.toList, filtered)
        val filteredWithBs = filtered.map{ case (interval, hypIndex) =>
          (interval, bs(hypIndex))          
        }
        //println(s"$interval, $nextInterval")
        //println(swept)
        for (h <- filtered.headOption) {
          val first = (interval, bs(h._2))
          accum append first
          accum ++ filtered.tail
        }
        sweepOverMinned(tail, accum)
      }
    }
  }
  
  val swept = Buffer[(Float, BleuStats)]()
  sweepOverMinned(minned, swept)
  val merged = swept.tail.foldLeft(Buffer[(Float, BleuStats)](swept.head)){  (accum, next) => 
    if (accum.last._2 != next._2 ) accum += next
    accum
  }
  println(merged)*/

}