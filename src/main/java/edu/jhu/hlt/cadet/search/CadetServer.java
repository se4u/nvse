package edu.jhu.hlt.cadet.search;

import static java.lang.System.exit;

import java.io.IOException;

import com.beust.jcommander.IStringConverter;
import org.apache.thrift.TException;
import org.apache.thrift.protocol.TCompactProtocol;
import org.apache.thrift.transport.TTransport;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;

import edu.jhu.hlt.concrete.services.search.SearchServiceWrapper;

public class CadetServer {
  private static Logger logger = LoggerFactory.getLogger(CadetServer.class);

  protected TTransport transport;
  protected TCompactProtocol protocol;
  public CadetEntitySearchProvider searchProvider;

  public static void main(String[] args) {
    Opts opts = new Opts();
    JCommander jc = null;
    try {
      jc = new JCommander(opts, args);
    } catch (ParameterException e) {
      System.err.println("Error: " + e.getMessage());
      exit(-1);
    }
    jc.setProgramName("./start.sh");
    if (opts.help) {
      jc.usage();
      return;
    }

    try {
      logger.info("Loading corpus");
      EntityFeatureCorpus efc = EntityFeatureCorpus.fromFile(opts.index);
      logger.info("Corpus loaded");
      CadetEntitySearchProvider prov = new CadetEntitySearchProvider(opts.languageCode, efc, opts);
      try (SearchServiceWrapper wrapper = new SearchServiceWrapper(prov, opts.port)) {
        logger.info("Preparing to run");
        wrapper.run();
      } catch (TException e) {
        logger.error("Unable to build search index", e);
        exit(-1);
      }
    } catch (ClassNotFoundException | IOException e1) {
      logger.error("Failed to load in passed in index file; is it the right file?", e1);
    }
  }
}
