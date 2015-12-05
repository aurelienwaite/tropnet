package com.sdl.tropnet

import java.io.File
import java.io.FileOutputStream
import java.io.FileWriter
import java.io.OutputStreamWriter
import java.io.Writer
import java.util.zip.GZIPOutputStream
import scala.io.Source
import resource._
import com.sdl.Hypothesis

object CreateFireVectors extends App {

  val nbestDir = args(0)
  val biasTerms = args(1).split(",").map(_.toDouble)
  val fireVecDir = args(2)

  def preprocessNBest(lines: Iterator[String], out: Writer) = {
    val ret = (for (line <- lines) yield {
      val fields = line.split("\t")
      val fVec = fields(1).split("[, ]").map(_.toDouble).toIndexedSeq
      (fields, fVec)
    }).toIndexedSeq
    //out.write("<s> </s>\t" + (List.fill(ret(0)._2.length *2){"0"}).mkString(",") + "\t0/999; 0/999; 0/999; 0/999; 999\t0.0\n")
    ret
  }

  def createFireVectors(fVec: IndexedSeq[Double], biases: (Double, Double)): TraversableOnce[IndexedSeq[Double]] = {
    val fire1 = -1.0d +: fVec.map(_ * biases._1)
    val fire2 = -1.0d +: fVec.map(_ * biases._2)
    val zeros = IndexedSeq.fill(fVec.length + 1) { 0.0d }
    Seq(fire1 ++ zeros,
      zeros ++ fire2,
      fire1 ++ fire2)
  }

  def createFireVectors(nbest: NBest, biases: (Double, Double)): NBest =
    for (h <- nbest; fire <- createFireVectors(h.fVec, biases)) yield Hypothesis(h.hyp, fire, h.bs, h.sbleu)

  for {
    nbestFile <- new File(nbestDir).listFiles
    in <- managed(Source.fromFile(nbestFile))
    out <- managed(new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(fireVecDir + nbestFile.getName + ".gz"))));
    hypRow <- preprocessNBest(Source.fromFile(nbestFile).getLines, out)
  } {
    val (fields, fVec) = hypRow
    for (
      newVec <- createFireVectors(fVec, (biasTerms(0), biasTerms(1)))
    ) {
      fields(1) = newVec.map(formatter.format(_)).mkString(",")
      out.write(fields.mkString("\t") + "\n")
    }
  }

}
