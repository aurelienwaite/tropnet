package com.sdl.tropnet

import java.io.FileWriter
import scala.util.Random
import resource._

object Main extends App{

  val outdir = args(0)

  val dim = args(1).toInt

  for (i <- 0 until 1000){
    val randStart = for (d <- 0 until dim) yield Random.nextGaussian
    for (out <- managed(new FileWriter(outdir +"/intitial" + i))){
      val iString = randStart.map(formatter.format(_)).mkString(" ")
      out.write(iString + "\n")
    }
  }

}
