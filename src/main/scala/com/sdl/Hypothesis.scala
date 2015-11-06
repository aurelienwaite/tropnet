package com.sdl

import com.sdl.BleuStats

case class Hypothesis(hyp : String, fVec : IndexedSeq[Double], bs : BleuStats, sbleu : Double){

  def withFVec(v : IndexedSeq[Double]) = Hypothesis(hyp, v, bs, sbleu)

}

object Hypothesis{

  def apply(line : String) : Hypothesis = {
    val fields = line.split("\t")
    val fVec = fields(1).split(",").map(_.toDouble).toIndexedSeq
    val sbleu = fields(3).toDouble
    Hypothesis(fields(0), fVec, BleuStats(fields(2)), sbleu)
  }

}
