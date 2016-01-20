package com.sdl.tropnet

import breeze.linalg.DenseVector
import com.sdl.NBest._
import java.io.File
import com.sdl.smert.SMERT
import breeze.linalg.DenseMatrix
import com.sdl.BleuStats

object Infer extends App {

  case class Config(
    nbestsDir: File = new File("."),
    params: Seq[DenseVector[Float]] = Nil,
    hyps: Boolean = false,
    stats: Boolean = false
  )

  def parseParams(paramString: String) = DenseVector(paramString.
      replace("unit:", "").split(",").map(_.toFloat))

  val parser = new scopt.OptionParser[Config]("smert") {
    head("smert", "1.0")
    //NBest Option
    opt[File]('n', "nbest_dir") required () action { (x, c) =>
      c.copy(nbestsDir = x)
    }
    arg[String]("parameter vectors...") unbounded() required () action { (x, c) =>
      c.copy(params = c.params :+ parseParams(x))
    }
    opt[Unit]('h', "hyps") action { (_, c) => 
      c.copy(hyps = true)
    }
    opt[Unit]('s', "stats") action { (_, c) => 
      c.copy(stats = true)
    }
  }

  val cliConf = parser.parse(args, Config()).getOrElse(sys.exit())
  import cliConf._
  
  assert(!params.isEmpty, "Need at least one parameter") 
  
  val topScoring = for{
    nbest <-loadUCamNBest(nbestsDir)
    mat = SMERT.nbestToMatrix(nbest)
  } yield {
    val (scores, activated) = if(params.size > 1) {
      val ones = DenseMatrix.ones[Float](1, mat.cols)
      val withBias = DenseMatrix.vertcat(ones, mat)
      val activated = for (p <- params) yield (p.t * withBias).t.map(math.max(_,0))
      ((activated.reduce(_ + _)).toArray, activated)
    } else {
      val scores = (params.head.t * mat).t
      (scores.toArray, Seq(scores))
    }
    val (maxScore, maxIndex) = scores.view.zipWithIndex.maxBy(_._1)
    val activatedStats = activated.map(a => if(a(maxIndex) ==0.0) 0 else 1)
    val res = if (maxScore == 0.0) 
      (0.0, 0, nbest(0).bs, activatedStats)
    else 
      (maxScore, maxIndex, nbest(maxIndex).bs,activatedStats)
    res
  }
  if(hyps)
    for(t<-topScoring) println(t._2)
  else if(stats){
    val activations = topScoring.map(_._4)
    val activatedStats = activations.reduce{(a,b) =>
      for((aActivated,bActivated) <- a zip b) yield aActivated + bActivated
    }
    for((count, i) <- activatedStats.view.zipWithIndex) println(s"Activations for unit $i: $count")
  }else {
    val aggregated = topScoring.map(_._3).reduce(_ + _)
    println(aggregated.computeBleu())
    println((for(nbest <-loadUCamNBest(nbestsDir)) yield nbest(0).bs).reduce(_ + _).computeBleu())
  }
}
