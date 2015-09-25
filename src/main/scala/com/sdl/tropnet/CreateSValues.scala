package com.sdl.tropnet

object CreateSValues extends App{

  val range = (1 to 4).reverse

  for (i <- range) {
    val s = 1.0/math.pow(2, i)
    println(s"1,$s")
  }

  for (i <- range) {
    val s = -1.0/math.pow(2, i)
    println(s"1,$s")
  }

}
