package edu.jhu.hlt.cadet.search;

import com.beust.jcommander.IStringConverter;
import com.beust.jcommander.ParameterException;

public class AlgorithmConverter implements IStringConverter<Algorithm> {
  @Override
  public Algorithm convert(String value) {
    Algorithm convertedValue = Algorithm.fromString(value);
    if(convertedValue == null) {
      throw new ParameterException("Value " + value + "can not be converted to Algorithm. " +
        "Available values are: binary, gaussian, neural.");
    }
    return convertedValue;
  }
}
