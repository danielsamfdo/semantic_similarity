def dict_dotprod(d1, d2):
  """Return the dot product (aka inner product) of two vectors, where each is
  represented as a dictionary of {index: weight} pairs, where indexes are any
  keys, potentially strings.  If a key does not exist in a dictionary, its
  value is assumed to be zero."""
  smaller = d1 if len(d1)<len(d2) else d2  # BUGFIXED 20151012
  total = 0
  for key in smaller.iterkeys():
      total += d1.get(key,0) * d2.get(key,0)
  return total