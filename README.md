# python-histogram - a lightweight Python histogram

Before getting carried away ... consider using numpy or any of the other
real statistical packages in python.

However, this might be useful for very simple/lightweight histogram needs.

A Histogram keeps track of how many times a given data point appears.
It is very much like a collections.Counter() with some extras.

TERMINOLOGY:
 * An "x value" is a data point being counted.
 * A "count" is the number of times a given x value was recorded.

EXAMPLE:

    h = Histogram()
    h.record(1)
    h.record(5)
    h.record(9)
    h.record(1)

The result is there are three distinct x values: 1, 5, 9.
The count for x=1 is 2; the counts for x=5 and x=9 are both 1.

## x values
In general x values should be arithmetic, but ANY hashable python data
type can be used with some caveats. For example, here is a histogram
of characters in a string:

    h = Histogram()
    for c in "we all live in a yellow submarine":
        h.record(c)

But CAVEAT: some stats or arithmetic methods will work, some won't:

    h.median_tuple()      -->   ('i', 'i')

however:

    h.weightedaverage()   -->   TypeError exception

See also MappedHistogram for better non-numeric data handling


