package com.sdl.tropnet

import com.sdl.NBest._

import java.io.File
import java.io.FileWriter
import scala.io.Source

import scala.language.postfixOps

object Scorer extends App{

  def bleuToString( bleu : (Double, Double) ) : String= {
    val (score, bp ) = bleu
    val points = score *100
    f"BLEU: $points%.3f [$bp%.4f]"
  }

  val nbestDir = args(0)
  val paramString = if (args.length > 1) Option(args(1)) else None
  val param = for(p <- paramString) yield {
    val parsed = p.split(",").map(_.toDouble)
    println("Using params: " + parsed.toIndexedSeq)
    parsed.map(_ * -1)
  }
  val fileoutputName = if (args.length > 2) Option(args(2)) else None
  val hypOut = for(name <- fileoutputName) yield new FileWriter(name)


  val bleuStats = for (nBest <- loadUCamNBest(new File(nbestDir))) yield {
    val oneBest = if(param.isEmpty) nBest(0) else{
      val scores = (for((hyp, i) <- nBest.view.zipWithIndex) yield {
        (hyp, hyp.fVec dot param.get, i)
      })
      val rescored = scores.sortWith(_._2 > _._2)
      for (out <- hypOut) out.write(rescored(0)._3 + "\n")
      rescored(0)._1
    }
    val oracle = nBest.sortWith(_.sbleu > _.sbleu)(0).bs
    (oneBest.bs, oracle)
  }
  val reduced = bleuStats.reduce( (t1, t2) => (t1._1 + t2._1, t1._2 + t2._2))

  for (out <- hypOut) out.close
  println("One Best " + bleuToString(reduced._1.computeBleu))
  println("Oracle " + bleuToString(reduced._2.computeBleu))

}
