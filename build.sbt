name := "tropnet"

version := "1.0"

organization := "University of Cambridge"

scalaVersion := "2.11.7"

scalacOptions += "-feature"

enablePlugins(JavaAppPackaging)

mainClass in Compile := Some("com.sdl.tropnet.CreateFireVectors")

mainClass in assembly := Some("com.example.Main")

libraryDependencies ++= Seq(
	"com.jsuereth" %% "scala-arm" % "1.4",
	"org.apache.spark" %% "spark-mllib" % "1.6.1" % "provided", 
	"org.scalatest" %% "scalatest" % "2.2.4" % "test",
	"org.scalanlp" %% "breeze-natives" % "0.11.2",
	"com.github.scopt" %% "scopt" % "3.3.0"
	)
