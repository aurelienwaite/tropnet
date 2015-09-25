package com.sdl.tropnet

import java.io.File
import scala.io.Source

import resource._

object SelectNonNorm extends App{

  val noNormDir = new File(args(0))
  val selectionFile = args(1)

  val selection = Source.fromFile(selectionFile).getLines.map(_.toInt).toSeq

  val files = noNormDir.listFiles.toSeq.sortWith(f2i(_) < f2i(_))
  //for (f <- files) println(f.getName)

  for ((i, nbestFile) <- selection zip files; s <- managed(Source.fromFile(nbestFile))) {
    val hyps = s.getLines.toIndexedSeq
    println(hyps(i/3))
  }
}
