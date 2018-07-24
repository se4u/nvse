package edu.jhu.hlt.cadet.search;

/**
 * Created by rastogi on 9/22/17.
 */

import com.beust.jcommander.Parameter;

public class Opts {
  @Parameter(names = { "--port", "-p" }, description = "The port the server will listen on.")
  int port = 8888;

  @Parameter(names = { "--index", "-i" }, description = "Path to the serialized index file.")
  String index = "data_dir/tac2017.pmf.binary".replaceFirst("^~",
    System.getProperty("user.home"));

  @Parameter(names={"--sentenceFile", "-s"}, description = "The file that contains the sentences.")
  String sFile = "data_dir/tac2017/entities.sentences";

  @Parameter(names={"--kQuery", "-q"}, description = "Maximum number of results to return to explain a query.")
  Integer kQuery = 10;

  @Parameter(names={"--kRationale", "-r"}, description = "Maximum number of rationale sentences to return to explain why a item was returned.")
  Integer kRationale = 10;

  @Parameter(names = {"--algorithm", "-a"}, converter = AlgorithmConverter.class, description = "Type of algorithm to use. (binary|gaussian|neural)")
  Algorithm algorithm = Algorithm.BINARY;

  @Parameter(names = { "--language", "-l" }, description = "The ISO 639-2/T three letter language code for corpus.")
  String languageCode = "eng";

  @Parameter(names = { "--help", "-h" }, help = true, description = "Print the usage information and exit.")
  boolean help;
}
