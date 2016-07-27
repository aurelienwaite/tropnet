package com.sdl

case class Hypothesis(hyp : String, fVec : IndexedSeq[Double], bs : BleuStats, sbleu : Double){

  def withFVec(v : IndexedSeq[Double]) = copy(fVec = v)
  
  override def toString() = {
    val fVecString = fVec.map{f => 
      if
        (f == Math.rint(f)) f.toInt.toString()
      else
        f"$f%6e"
    }.mkString(",")
    f"$hyp%s\t$fVecString%s\t$bs%s\t$sbleu%.6f"
  }

}

object Hypothesis{

  def apply(line : String) : Hypothesis = {
    val fields = line.split("\t")
    val fVec = fields(1).split(",").map(_.toDouble).toIndexedSeq
    val sbleu = fields(3).toDouble
    Hypothesis(fields(0), fVec, BleuStats(fields(2)), sbleu)
  }
  

}
