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
    params1: DenseVector[Float] = DenseVector(),
    params2: DenseVector[Float] = DenseVector(),
    biases: (Float, Float) = (0,0),
    hyps: Boolean = false
  )

  def parseParams(paramString: String) = DenseVector(paramString.split(",").map(_.toFloat))

  val parser = new scopt.OptionParser[Config]("smert") {
    head("smert", "1.0")
    //NBest Option
    opt[File]('n', "nbest_dir") required () action { (x, c) =>
      c.copy(nbestsDir = x)
    }
    opt[String]('1', "1st_weights") required () action { (x, c) =>
      c.copy(params1 = parseParams(x))
    }
    opt[String]('2', "2nd_weights") required () action { (x, c) =>
      c.copy(params2 = parseParams(x))
    }
    opt[Unit]('h', "hyps") action { (_, c) => 
      c.copy(hyps = true)
    }
  }

  val cliConf = parser.parse(args, Config()).getOrElse(sys.exit())
  import cliConf._
  
  //println(loadUCamNBest(nbestsDir).size)
  
  val topScoring = for{
    nbest <-loadUCamNBest(nbestsDir)
    mat = SMERT.nbestToMatrix(nbest)
  } yield {
    val ones = DenseMatrix.ones[Float](1, mat.cols)
    val withBias = DenseMatrix.vertcat(ones, mat)
    /*for(s <- (params1.t * withBias).t) {
      println(s)
    }
    sys.exit*/
    val scores1 = (params1.t * withBias).t.map(math.max(_,0))
    val scores2 = (params2.t * withBias).t.map(math.max(_,0))
    val scores = (scores1 + scores2).toArray
    val (maxScore, maxIndex) = scores.view.zipWithIndex.maxBy(_._1)
    val res = if (maxScore == 0.0) 
      (0.0, 0, nbest(0).bs)
    else 
      (maxScore, maxIndex, nbest(maxIndex).bs)
    //println(s"${max._1} $res")
    res
  }
  if(hyps)
    for(t<-topScoring) println(t._2)
  else {
    val aggregated = topScoring.map(_._3).reduce(_ + _)
    println(aggregated.computeBleu())
    println((for(nbest <-loadUCamNBest(nbestsDir)) yield nbest(0).bs).reduce(_ + _).computeBleu())
  }
}
