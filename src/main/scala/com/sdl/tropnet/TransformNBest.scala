package com.sdl.tropnet

import java.io.File
import java.io.FileInputStream
import java.io.FileWriter
import java.util.zip.GZIPInputStream
import resource._
import scala.io.Source
import com.sdl.Hypothesis

object TransformNBest extends App{

  def transformHyp(hyp : Hypothesis, index : Int)={
    val fVec = hyp.fVec
    val transformedFVec = (for((fea, i) <- fVec.view.zipWithIndex) yield {
      val negated = if (fea == 0) fea else formatter.format(fea.toDouble * -1)
      f"tropnet$i%d:::TrainingData $negated"
    }).mkString(" ")
    val bs = hyp.bs
    val bsString = ((for (hit <- bs.hits; stat <-hit.productIterator) yield stat.toString) :+ bs.refLength.toString).mkString(" ")
    List(index.toString, bsString, transformedFVec).mkString(" ||| ")
  }

  val nbestDir = args(0)
  val output = args(1)


  for {out <- managed(new FileWriter(output))
    (nbestFile, i) <- new File(nbestDir).listFiles.view.zipWithIndex
    in <- managed(Source.fromInputStream(if (nbestFile.getName.endsWith(".gz"))
      new GZIPInputStream(new FileInputStream(nbestFile))
      else new FileInputStream(nbestFile)))
    hypRow <- in.getLines
  } out.write(transformHyp(Hypothesis(hypRow), i) +"\n")
}
