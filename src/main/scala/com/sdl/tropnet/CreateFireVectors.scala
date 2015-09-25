package com.sdl.tropnet

import java.io.File
import java.io.FileOutputStream
import java.io.FileWriter
import java.io.OutputStreamWriter
import java.io.Writer
import java.util.zip.GZIPOutputStream
import scala.io.Source

import resource._

object CreateFireVectors extends App{

  val nbestDir = args(0)
  val biasTerms = args(1).split(",").map(_.toDouble)
  val fireVecDir = args(2)

  def preprocessNBest(lines : Iterator[String], out : Writer) ={
    val ret = (for (line <- lines) yield {
      val fields = line.split("\t")
      val fVec = -1.0 :: fields(1).split("[, ]").map(_.toDouble).toList
      (fields, fVec)
    }).toIndexedSeq
    //out.write("<s> </s>\t" + (List.fill(ret(0)._2.length *2){"0"}).mkString(",") + "\t0/999; 0/999; 0/999; 0/999; 999\t0.0\n")
    ret
  }

  for {nbestFile <- new File(nbestDir).listFiles
    in <- managed(Source.fromFile(nbestFile))
    out <- managed(new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(fireVecDir + nbestFile.getName +".gz"))));
    hypRow <- preprocessNBest(Source.fromFile(nbestFile).getLines, out)
  } {
    val (fields, fVec) = hypRow
    val fire1 = fVec.map(_ * biasTerms(0))
    val fire2 = fVec.map(_ * biasTerms(1))
    val zeros = List.fill(fVec.length) { 0.0d }
    for (newVec <- List(fire1 ::: zeros,
      zeros ::: fire2,
      fire1 ::: fire2)){
      fields(1) = newVec.map(formatter.format(_)).mkString(",")
      out.write(fields.mkString("\t") +"\n")
    }
  }
}
