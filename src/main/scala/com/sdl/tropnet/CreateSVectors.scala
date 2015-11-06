package com.sdl.tropnet

import com.sdl.NBest._

import java.io.File
import java.io.FileOutputStream
import java.io.FileWriter
import java.io.OutputStreamWriter
import java.util.zip.GZIPOutputStream
import resource._


object CreateSVectors extends App{

  val nbestDir = args(0)
  val hiddenTerms = args(1).split(",").map(_.toDouble).toSeq
  val output = args(2)

  for {
    out <- managed(new FileWriter(output))
    (nbest, i) <- loadUCamNBest(new File(nbestDir)).view.zipWithIndex
    hyp <- nbest
  } {
    val fireVec = -1.0 +: hyp.fVec
    val s1 = hiddenTerms.slice(0, fireVec.length) dot fireVec
    val s2 = hiddenTerms.slice(fireVec.length, hiddenTerms.length) dot fireVec
    for(s <- List(IndexedSeq(s1,0), IndexedSeq(0,s2), IndexedSeq(s1,s2))){
      val sHyp = hyp.withFVec(s)
      out.write(TransformNBest.transformHyp(sHyp,i) + "\n")
    }
  }

}
