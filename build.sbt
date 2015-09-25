name := "tropnet"

version := "1.0"

organization := "University of Cambridge"

scalaVersion := "2.11.7"

scalacOptions += "-feature"

enablePlugins(JavaAppPackaging)

mainClass in Compile := Some("com.sdl.tropnet.CreateFireVectors")

libraryDependencies ++= Seq("com.jsuereth" %% "scala-arm" % "1.4")
