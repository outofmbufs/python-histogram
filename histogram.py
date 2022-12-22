# Copyright (c) 2022 Neil Webber
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import collections.abc
from collections import Counter


# USE NUMPY! ... Before going too far with this, consider numpy instead.
# But for very simple/lightweight histogram needs, this might be useful.
#
# A Histogram keeps track of how many times a given data point appears.
# It is very much like a collections.Counter() with some extras.
#
# TERMINOLOGY:
#     An "x value" is a data point being counted.
#     A "count" is the number of times a given x value was recorded.
#
# EXAMPLE:
#     h = Histogram()
#     h.record(1)
#     h.record(5)
#     h.record(9)
#     h.record(1)
#
# There are three distinct x values: 1, 5, 9.
# The count for x=1 is 2; the counts for x=5 and x=9 are both 1.
#
# In general x values should be arithmetic, but ANY hashable python data
# type can be used with some caveats. For example, here is a histogram
# of characters in a string:
#
#     h = Histogram()
#     for c in "we all live in a yellow submarine":
#         h.record(c)
#
# But CAVEAT: some stats or arithmetic methods will work, some won't:
#
#         h.median_tuple()      -->   ('i', 'i')
# but:    h.weightedaverage()   -->   TypeError exception
#
# See also MappedHistogram for better non-numeric data handling
#
# IMPLEMENTATION NOTE: Implementing this as a dict subclass (rather than
# as a Counter subclass) and manually providing the "counts start at zero"
# logic in record() was substantially faster for record() operations. The
# assumption is that if anything is performance critical, record() would
# be the one that matters.
#
class Histogram(dict):
    """A Histogram counts occurrences of specific data points.

    Each recorded data point is called an "x value" or just "x".

    An application may want to treat ranges of multiple x values
    as equivalent for counting purposes; see MappedHistogram for that.

    If h is a Histogram:
       h.record(x)    -- adds one to the count of occurrences of x.
       h[x]           -- h.__getitem__(x): returns the count of x.
                         NOTE: Raises KeyError for ANY x that has a zero
                               count. SEE ALSO: "in" / __contains__()
       x in h         -- h.__contains__(x): True if x has a non-zero count.
       h.n            -- sum of all counts
       h.elements()   -- Like Counter.elements.
                         If h[x] == N, this generates 'x' N times.
                         NOTE: not in any particular order.
       h.clear()      -- reset histogram to initial state.

    Iterating a histogram returns each x value with a non-zero count (once).
    Example:
        for x in h:
            print(f"At least one {x} was recorded")
    """

    def __init__(self):
        """Initialize a histogram. There are no arguments: Histogram()"""
        self.clear()

    def __repr__(self):
        return f"<{self.__class__.__qualname__}() n={self.n} @ 0x{id(self):x}>"

    def record(self, x, /, *, n=1):
        """Record one or more (default: n=1) occurrences of a value (x).

        Will 'unrecord' x if n<0.
        """

        # NOTE: This code is careful to make the standard case of simply
        #       adding to an existing bucket count be as fast as possible.
        #       It's a try, a +=, and the test for n < 0; under which all
        #       special cases hide (also in the KeyError for first time)
        try:
            self[x] += n
        except KeyError:
            if n < 0:
                raise ValueError(f"Can't unrecord {x}; never recorded.")
            elif n > 0:
                self[x] = n

        # Note that the above carefully kept zero entries from
        # ever being created. If this is an unrecord, figure out:
        #    - if it becomes zero, delete it
        #    - if it is too much being unrecorded, undo it and complain
        if n < 0:
            if self[x] == 0:
                del self[x]
            elif self[x] < 0:
                self[x] -= n
                raise ValueError(f"record({x}, n={n}) but [{x}]={self[x]}")

    def __iter__(self):
        """Histogram iterator: generates sorted (low-to-high) x values."""
        yield from sorted(super().__iter__())

    # like dictionary.get - but default value is zero rather than None
    def get(self, key, default=0):
        try:
            return self[key]
        except KeyError:
            return default

    def elements(self):
        """Generate all the recorded x values.

        In contrast to the iterable, this returns each x value however
        many times each x was recorded (whereas the normal iterator returns
        each unique x value only once)
        """
        return Counter(self).elements()

    def most_common(self, *args, **kwargs):
        """Return list of x values and their counts, ordered by count.

        Semantically identical to Counter(self).most_common()
        """
        return Counter(self).most_common(*args, **kwargs)

    @property
    def n(self):
        """Return the total number of values recorded."""
        return sum(self.values())   # ".values()" are the counts

    def weightedaverage(self):
        """Return the weighted average of all recorded x values."""
        total = 0
        for x in self:
            total += x * self[x]
        return total/self.n

    def median_tuple(self):
        """Return a TUPLE (x1, x2) representing the median x value.

        The return value will ALWAYS be a tuple:
               (x1, x2)
        where both x1 and x2 are x values that are in the histogram
        (i.e., have appeared in a record() call).

        The mathematical median lies at m = (x1 + x2) / 2

        If median is itself a x value, x1==x2, but a 2-element tuple
        is still returned (a 2-element tuple is ALWAYS returned).
        """

        sorted_xvals = list(self)

        # Degenerate cases
        if len(sorted_xvals) == 0:
            raise ValueError("empty histogram: no median")
        elif len(sorted_xvals) == 1:   # N (>=1) data points, but all the same
            return (sorted_xvals[0], sorted_xvals[0])

        total = self.n

        # the median is the one in the middle, or a pair in the middle
        med_nth = (total + 1) // 2

        # take each bin in sorted order, and count the data points until
        # the bin that is the median (med_nth) one

        count = 0
        for xi, x in enumerate(sorted_xvals):
            if count + self[x] >= med_nth:
                break
            count += self[x]

        # this bin (x) is the median ...
        m_tuple = (x, x)

        # but if total was even then there is no single "center" data point.
        # Check to see if need a different x2.
        if (total % 2) == 0:
            # if there aren't more values in this bin, need next x2
            if count + self[x] == med_nth:   # i.e., no more here
                m_tuple = (x, sorted_xvals[xi+1])
        return m_tuple

    def median(self):
        """Return the arithmetic median, which might not be an x value."""
        return sum(self.median_tuple()) / 2

    def modes(self, n_down=0):
        """Return a list of the x values that appear the most often.

        The return value is ALWAYS a list, even if there is just one mode.
        Multiple (identical count) modes are in order from low to high.

        Argument n_down asks for the "nth" highest mode, with 0 being the
        top-most (standard mode definition). For example, n_down=1 ignores
        the standard mode and returns the next-highest mode x value(s).
        """

        mc = self.most_common()
        while n_down >= 0:
            if not mc:
                return []
            x, mval = mc.pop(0)
            results = [x]

            # extend the list with all x values that have this same mval count
            while mc and mval == mc[0][1]:
                x, _ = mc.pop(0)
                results.append(x)
            n_down -= 1

        return sorted(results)

    def cumulative(self):
        """Return a sorted (x low-to-high) list of tuples (x, cm).

        Each cm is the cumulative count from the first
        x to/including this one."""

        running = 0
        return [(x, running := running + self[x]) for x in self]

    #
    # This generates tuples: (x, str) where "str" will consist of
    # N characters ("char") consistent with a bar graph of counts for x.
    # To print a primitive (unlabeled) chart:
    #       for x, s in hist.genstrings():
    #           print(s)
    # If fullscalevalue is given, that will be the value which will result
    # in a maxwidth string (anything over this value will get truncated).
    # By default, fullscalevalue is taken from the max count in the histogram.
    #
    def genstrings(self, maxwidth=75, fullscalevalue=None, char="*"):
        """Generate tuples suitable for using to print an ASCII depiction."""
        if fullscalevalue is None:
            fullscalevalue = self[self.modes()[0]]
        scale = fullscalevalue/maxwidth
        for x in self:
            s = char * int(self[x]/scale)
            s = ("{:."+str(maxwidth)+"s}").format(s)
            yield (x, s)


#
# A RangedHistogram discards points above/below xmax/xmin (if given).
# Obviously if neither is given a RangedHistogram is a plain Histogram.
#
# Points outside the limits are not recorded in the histogram itself.
# They are tracked as "overs" or "unders" accordingly. They will not be
# any part of any computed data (median, etc).
#
# If xmin is:
#   * None: No minimum limit will be imposed.
#   * A callable object: it will be invoked with one argument, the data
#     point in question. If it returns True, the data point is an "unders"
#     and will not be recorded.
#   * Anything else: xmin will be used as if this lambda
#                lambda x: x < xmin
#     had been given instead of a naked value.
#
# xmax is treated analogously.
#
class RangedHistogram(Histogram):
    """A RangedHistogram is a Histogram with enforced min/max for x values."""

    def __init__(self, *, xmin=None, xmax=None, **kwargs):
        """RangedHistogram(*, xmin=None, xmax=None)"""
        super().__init__(**kwargs)
        self.overs = 0
        self.unders = 0
        self.xmin = xmin
        self.xmax = xmax

        # convert xmin/xmax None into an always-false callable
        # NOTE: fxmin/fxmax might still be a naked value (not callable)
        fxmin = (lambda x: False) if xmin is None else xmin
        fxmax = (lambda x: False) if xmax is None else xmax

        # convert naked values into appropriate function
        self._fxmin = fxmin if callable(fxmin) else lambda x: x < self.xmin
        self._fxmax = fxmax if callable(fxmax) else lambda x: x > self.xmax

    def clear(self):
        """Reset a RangedHistogram to its original no-data state."""
        super().clear()
        self.overs = 0
        self.unders = 0

    def __repr__(self):
        comma = ""
        s = f"<{self.__class__.__qualname__}("
        if self.xmin is not None:
            s += f"xmin={self.xmin}"
            comma = ", "
        if self.xmax is not None:
            s += comma + f"xmax={self.xmax}"
        return s + f") n={self.n} @ 0x{id(self):x}>"

    def record(self, x, /, *, n=1):
        """Record one or more (default: n=1) occurrences of a value (x)."""
        if self._fxmin(x):
            self.unders += n
        elif self._fxmax(x):
            self.overs += n
        else:
            super().record(x, n=n)

        if n < 0:     # check to make sure didn't unrecord more than recorded
            if self.unders < 0 or self.overs < 0:
                # undo whichever was in error
                if self.unders < 0:
                    self.unders -= n
                else:
                    self.overs -= n
                raise ValueError("unrecorded more overs/unders than recorded")


#
# A MappedHistogram is a subclass of RangedHistogram with a mapping function
# that transforms x values before they are recorded.
#
# WHAT IS A MAPPING FUNCTION
#
# A mapping function controls how many "bins" will be used to count data
# values, and how nearby/adjacent data values are collapsed together into
# those bins (and/or any other useful transformation).
#
# Example: count byte values from 0 .. 255 but only count
# how they fall into four bins:
#
#      0 .. 63
#      64 .. 127
#      128 .. 191
#      192 .. 255
#
# This is done with a mapping function that converts the input values into
# one of four values. The "best" way to do this is usually to convert every
# input to the midpoint of the appropriate bin.
#
# So, for example, this mapping:
#
#      mapped_x = 32 + (64 * (x // 64))
#
# converts every value between 0 and 255 into 32, 96, 160, or 224; those
# "canonical bin values" are then recorded in the histogram.
#
# Many mappings require esablishing xmin/xmax as well as a mapping function
# but the details are mapping- and application-specific.
#
# DEFINING MAPPINGS
#
# Custom mappings are defined by defining a mapper() method in a subclass.
# NO mapper() is provided in MappedHistogram(); subclasses MUST define one.
#
# A NOTE ABOUT "GOOD" MAPPING FUNCTIONS
#
# Good mapping functions are "stable", defined as:
#    Let f() be the mapping function
#
#    If f(f(x)) == f(x), the mapping is stable.
#
# In english - a mapping is stable if putting a mapped x value through
# the mapping again returns the same mapped value.
#
# Unstable mappings will generally "work" but with caveats.  Mapped x values
# leak back out via iterators, elements(), and the stats methods.
# With an unstable mapping, if mapped x values are submitted a
# second time the results are unlikely to be desirable.
#
# Applications that need an unstable mapping should consider instead
# explicitly managing the from/to mapping operations outside the histogram.
#

class MappedHistogram(RangedHistogram):
    """Abstract class for mapped histograms.

    MUST BE SUBCLASSED and the subclass must define a mapper() method.
    """

    def record(self, x, /, *, n=1):
        """Record one or more x values, which will be mapped."""
        super().record(self.mapper(x), n=n)

    def __getitem__(self, index):
        """Return count for index, which will be mapped: self.mapper(index)."""
        return super().__getitem__(self.mapper(index))

    # Note that the definition of 'in' for MappedHistogram is that
    # the **mapped value** of item is 'in'.
    def __contains__(self, item):
        return super().__contains__(self.mapper(item))

    def get(self, key, default=0):
        return super().get(self.mapper(key), default=default)


# This subclass implements a simple modulus histogram.
# For example:
#     h = ModuloHistogram(modulus=2)
# creates a histogram that will basically count even vs odd data points.
#
class ModuloHistogram(MappedHistogram):
    def __init__(self, *, modulus, **kwargs):
        super().__init__(**kwargs)
        if modulus < 1 or int(modulus) != modulus:
            raise ValueError(f"modulus ({modulus}) must be int and > 1")
        self.modulus = modulus

    def mapper(self, x):
        return x % self.modulus


# This subclass expects x values to be integer values in range xmin to xmax.
# That range is divided into the specified number of buckets.
# Values outside that range will be overs/unders per RangedHistogram.
#
# NOTE: Integer arithmetic is used to compute the bin widths (sizes) and
#       scaling; consequently, BEST PRACTICE is nbins should evenly
#       divide (xmax - xmin + 1). Or, if that is not possible, should not
#       be "too big" compared to (xmax - xmin + 1). Ignoring these guidelines
#       may result in fewer bins than requested, because, for example, if
#       the bin width is computed to be 1.x it has to be rounded up to 2.
#
# For example:
#   Works great:
#     IntRangeHistogram(xmin=0, xmax=255, nbins=4)
#   Works ok - but last bin will be smaller than the others:
#     IntRangeHistogram(xmin=0, xmax=255, nbins=5)
#   Likely to not have desired effect:
#     IntRangeHistogram(xmin=0, xmax=255, nbins=200)
#
# NOTE: By default the midpoint (i.e., canonical values) will be integer.
#       Specify intmid=False to get a floating mapped value instead.
#
class IntRangeHistogram(MappedHistogram):
    """Histogram that maps integer range into fixed buckets."""

    def __init__(self, *, nbins, xmin, xmax, intmid=True, **kwargs):
        super().__init__(xmin=xmin, xmax=xmax, **kwargs)

        if int(nbins) != nbins:
            raise ValueError(f"nbins ({nbins}) must be integer")

        # total number of possible integer values
        nvalues = xmax - xmin + 1

        # there must be fewer bins than values or the math doesn't work.
        # Said another way - each bin must span more than just 1 unit of value.
        if nbins >= nvalues:
            raise ValueError(f"{nbins} bins but only {nvalues} in range.")

        # Compute bin width needed to the entire range of possible values.
        # Round up if nvalues is not a multiple of nbins, in which case the
        # last bin sort of goes past xmax (but xmax still limits the range).
        self.binw = (nvalues + (nbins - 1)) // nbins

        # the half bin-width is used for computing the canonical (mid-bin)
        # value associated with each bin.
        self.halfbw = self.binw / 2
        if intmid:
            self.halfbw = int(self.halfbw)

    def mapper(self, x):
        bin_number = (x - self.xmin) // self.binw
        return self.xmin + self.halfbw + (bin_number * self.binw)


# Treats [xmin, xmax] as a real-number CLOSED interval and divides
# that into 'nbins' bins.
#
# NOTE: The fact that both xmin and xmax are in the range presents
# a floating point "1 ULP" problem - Putting xmax into the top
# makes that bin 1 ULP wider than it "should" be. All other
# bins are semi-OPEN [bin-bottom, bin-top)
#
class RealRangeHistogram(MappedHistogram):
    """Histogram that maps a real range into fixed buckets."""
    def __init__(self, *, nbins, xmin, xmax, **kwargs):
        super().__init__(xmin=xmin, xmax=xmax, **kwargs)
        if int(nbins) != nbins:
            raise ValueError(f"nbins ({nbins}) must be integer")

        self.nbins = nbins
        self.binsize = (xmax - xmin)/nbins
        self.baseoffset = xmin + (self.binsize/2)

    def mapper(self, x):
        if x == self.xmax:    # this is where the last bin is "1 ULP" too big
            binnumber = self.nbins - 1
        else:
            binnumber = int((x - self.xmin) / self.binsize)
        return self.baseoffset + (self.binsize * binnumber)


# Generic "bin mapping" histogram
#
# A binmap is an UNORDERED collection of COMPARISONMAPs.
#
# Each COMPARISONMAP is a tuple of two elements:
#      * A sequence of COMPARISONCONDITIONs (func, arg) that will each
#        be applied to the value 'x' being tested: func(x, arg)
#      * A mapped value that will be used if all the COMPARISONCONDITIONS
#        are true
#
# EXAMPLE
#  Map all values that are:
#          >= -1.0 and < 0 to -0.5
#          <= 1.0 and > 0 to 0.5
#          == 0 to 0
#
#  binmap = [
#              ([(operator.ge, -1.0), (operator.lt, 0)], -0.5),
#              ([(operator.le, 1.0), (operator.gt, 0)], 0.5),
#              ([(operator.eq, 0.0)], 0)
#           ]
#
# By default if no mapping is found, ValueError is raised. This can be
# overridden by supplying a nomatch_f argument, which MUST BE A CALLABLE.
# It will be invoked as:
#      nomatch_f(x)
# and the value it returns will be used as the mapped value. A simple lambda
# can be used to always return a constant, e.g.,
#      nomatch_f=lambda x: 17
#
# IMPORTANT NOTE ABOUT COMPARISONMAP ORDERING
#
# The mapping function caches COMPARISONMAPs and does not necessarily
# apply them in the order they appear in the binmap. Therefore, behavior
# is UNDEFINED if a given 'x' value will satisfy more than one COMPARISONMAP
#

class BinMapHistogram(MappedHistogram):

    def __init__(self, *, binmap, nomatch_f=None, **kwargs):
        super().__init__(**kwargs)

        if nomatch_f is not None and not callable(nomatch_f):
            raise TypeError(f"nomatch_f (={nomatch_f}) must be a callable")

        self.binmap = binmap
        self.nomatch_f = nomatch_f

    # CACHING POLICY: there's a tradeoff between rearranging the binmap
    # and looping through N entries. LRUTHRESH controls how deep into the
    # binmap a COMPARISONMAP has to be to be pulled to the front.
    # NOTE: Setting LRUTHRESH >= len(binmap) disables re-ordering.
    LRUTHRESH = 2

    # Further notes on caching:
    # The mapper necessarily has to search the binmap linearly.
    # If the data is truly randomly distributed then this search
    # will be roughly O(N) where N is the length of the binmap.
    #
    # On the assumption there might be *some* locality in the data
    # the binmap is managed in (near) LRU fashion, so that the more
    # recently-matched entries are towards the front. This has been
    # measured to improve performance (and not impose much overhead)
    # DEPENDING, OF COURSE, ON HOW MUCH DATA LOCALITY THERE IS. For
    # truly random data this is a bunch of work for little or no benefit.

    def mapper(self, x):

        for i, cm in enumerate(self.binmap):
            conditions, mappedx = cm
            if all(map(lambda c: c[0](x, c[1]), conditions)):
                if i > self.LRUTHRESH:
                    self.binmap = [cm] + self.binmap[:i] + self.binmap[i+1:]
                return mappedx

        # no COMPARISONMAP entry matched
        try:
            return self.nomatch_f(x)
        except TypeError:     # nomatch_f could be None...
            raise ValueError(f"no mapping for {x}") from None


# This mapper is for histogramming non-arithmetic values.
# It expects a dictionary ('dictmap') mapping keys to values.
#
# For convenience, if the dictmap is not a dictionary but can be
# iterated (more specifically: enumerate()), then the entries of the
# list will be taken as keys and list position is the mapped value.
#
# To achieve a stable mapping, if none of the resulting mapped values are
# themselves in the map, they will be automatically added to (a copy of)
# the map with an identity mapping, to fulfill:
#
#          Let f() be the mapping function
#          If f(f(x)) == f(x), the mapping is stable.
#
# This can be disabled by setting auto_stable=False (default is True)
#
# The keyword arg
#
#       defaultmapped=foo
#
# establishes foo as the default mapping for any x value not in the map.
#
class DictMapHistogram(MappedHistogram):
    __DEFAULTSENTINEL = object()

    def __init__(self, *, dictmap, auto_stable=True,
                 defaultmapped=__DEFAULTSENTINEL, **kwargs):
        super().__init__(**kwargs)

        # if not given a dict, make it into a dict
        if not isinstance(dictmap, collections.abc.Mapping):
            dictmap = {x: i for i, x in enumerate(dictmap)}

        self.dictmap = dict(dictmap)     # important to be a copy, see below
        self.auto_stable = auto_stable
        self.defaultmapped = defaultmapped
        self.use_default = (defaultmapped is not self.__DEFAULTSENTINEL)

        if auto_stable:
            # add any mapping values that are not already in the mapping
            # note: this iterates the *supplied* dictmap & alters the *copy*
            for k, v in dictmap.items():
                if v not in self.dictmap:
                    self.dictmap[v] = v

            if self.use_default and defaultmapped not in self.dictmap:
                self.dictmap[defaultmapped] = defaultmapped

    def mapper(self, x):
        try:
            return self.dictmap[x]
        except KeyError:
            if self.use_default:
                return self.defaultmapped
            else:
                raise


if __name__ == "__main__":
    import unittest
    import operator

    class TestMethods(unittest.TestCase):

        # these all make "simple" histograms but do it
        # from the various subclasses. Some of these require
        # test data be numeric and within the given limits.
        # The test code "just knows" this.
        # NOTE: The generic tests won't work with mapped histograms
        #       that don't preserve 1:1-ness of the test data.
        factories = (
            lambda: Histogram(),
            lambda: RangedHistogram(),
            lambda: RangedHistogram(xmin=lambda x: False,
                                    xmax=lambda x: False),
            lambda: RangedHistogram(xmin=-9999, xmax=9999),
            lambda: RangedHistogram(xmin=-9999),
            lambda: RangedHistogram(xmax=9999),
            # lambda: MappedHistogram(),
            lambda: ModuloHistogram(modulus=10000),
        )

        def test_h_basics(self):
            for factory in self.factories:
                with self.subTest(h=factory()):
                    # if any vals are negative the ModuloHistogram tests fail
                    vals = (10, 20, 30, 10, 20, 10, 20, 1, 1, 1, 9)
                    results = ((10, 3), (20, 3), (30, 1), (1, 3), (9, 1))
                    h = factory()
                    for x in vals:
                        h.record(x)
                    for x, count in results:
                        self.assertEqual(h[x], count)
                    self.assertEqual(h.n, len(vals))

                    # make sure everything in the histogram says is
                    # in it (i.e., the __iter__) is ... in it (__contains__)
                    for x in h:
                        self.assertTrue(x in h)

                    # make sure everything in the histogram came from a val
                    for x in h:
                        self.assertTrue(x in vals)

        def test_h_recordn(self):
            for factory in self.factories:
                h = factory()
                with self.subTest(h=h):
                    for i in range(1, 10):
                        h.record(i, n=i)
                    for i in range(1, 10):
                        self.assertEqual(h[i], i)

                    # recording zero should be a no-op, including not
                    # making an entry for the x value
                    h.record(123, n=0)
                    self.assertTrue(123 not in h)

        def test_h_unrecord(self):
            for factory in self.factories:
                h = factory()
                with self.subTest(h=h):
                    for i in range(1, 10):
                        h.record(i, n=i)
                    for i in range(1, 10):
                        h.record(i, n=-i)
                    for i in range(1, 10):
                        self.assertFalse(i in h)

                    # try unrecording too much
                    h.record(17)
                    with self.assertRaises(ValueError):
                        h.record(17, n=-2)
                    # after the failed "un"record, state should be unchanged:
                    self.assertEqual(h[17], 1)

                    # same test but with a non-existent entry to begin with
                    with self.assertRaises(ValueError):
                        h.record(1234, n=-2)
                    self.assertFalse(1234 in h)

        def test_h_get(self):
            for factory in self.factories:
                h = factory()
                with self.subTest(h=h):
                    h.record(17)
                    self.assertEqual(h.get(17), 1)
                    self.assertEqual(h.get(99), 0)

        def test_h_ex(self):
            for factory in self.factories:
                h = factory()
                with self.subTest(h=h):
                    with self.assertRaises(KeyError):
                        _ = h[17]

                    h.record(17)
                    h.record(17, n=-1)
                    with self.assertRaises(KeyError):
                        _ = h[17]

        def test_elements_and_byc_and_modes(self):
            for factory in self.factories:
                h = factory()
                for i in range(1, 10):
                    h.record(i, n=i)

                self.assertEqual(h.modes(), [9])

                for i in range(1, 10):
                    self.assertEqual(h.modes(n_down=i-1), [10-i])

                # this test case has two modes
                h2 = factory()
                h2.record(17, n=10)
                h2.record(42, n=10)
                h2.record(1)
                h2.record(30)
                h2.record(69)
                self.assertEqual(h2.modes(), [17, 42])

                c = Counter()
                for x in h.elements():
                    c[x] += 1

                for i in range(1, 10):
                    self.assertEqual(c[i], i)

                mc = h.most_common()
                for i in range(1, 10):
                    self.assertEqual(mc[i-1][0], 10 - i)

        def test_nonnumeric(self):
            # histogram some non-numeric data. Can't use all the factories
            # for this so just do something ad-hoc.
            vals = "we all live in a yellow submarine"
            counts = {c: vals.count(c) for c in vals}
            h = Histogram()
            for c in vals:
                h.record(c)
            for c, n in counts.items():
                self.assertEqual(h[c], n)
            self.assertEqual(h.n, len(vals))

            # surely overkill but...
            for c in h:
                self.assertTrue(c in vals)
            for c in vals:
                self.assertTrue(c in h)
            for c in [chr(i) for i in range(256)]:
                self.assertEqual(c in h, c in vals)

        def test_wavg(self):
            for factory in self.factories:
                h = factory()
                with self.subTest(h=h):
                    for i in range(1, 5):
                        h.record(i)
                    self.assertEqual(h.weightedaverage(), 2.5)  # hand computed
                    h.clear()
                    for i in range(1, 5):
                        h.record(i, n=i)
                    self.assertEqual(h.weightedaverage(), 3.0)  # hand computed

        def test_median(self):
            h = Histogram()
            self.assertRaises(ValueError, h.median_tuple)
            self.assertRaises(ValueError, h.median)

            h.record(1)
            self.assertEqual(h.median_tuple(), (1, 1))
            self.assertEqual(h.median(), 1)

            h.record(2)
            self.assertEqual(h.median_tuple(), (1, 2))
            self.assertEqual(h.median(), 1.5)

            h.record(99)
            self.assertEqual(h.median_tuple(), (2, 2))
            h.record(100)
            self.assertEqual(h.median_tuple(), (2, 99))
            h.record(2)
            self.assertEqual(h.median_tuple(), (2, 2))
            h.record(2, n=-1)
            h.record(100)
            self.assertEqual(h.median_tuple(), (99, 99))

        def test_cumulative(self):
            low_x = 1
            high_x = 25
            special_x = 17      # anywhere within low to high

            def _fill_it(h):
                for i in range(low_x, high_x + 1):
                    h.record(i, n=i)

            for factory in self.factories:
                h = factory()
                with self.subTest(h=h):
                    self.assertEqual(h.cumulative(), [])
                    _fill_it(h)
                    cm = h.cumulative()
                    running = 0
                    for i in range(low_x, high_x + 1):
                        x, c = cm.pop(0)
                        running += i
                        with self.subTest(x=x, c=c, i=i, running=running):
                            self.assertEqual(x, i)
                            self.assertEqual(c, running)

                    for bump in (low_x, special_x, high_x):
                        h = factory()
                        _fill_it(h)
                        h.record(bump)
                        cm = h.cumulative()
                        running = 0
                        for i in range(low_x, high_x + 1):
                            x, c = cm.pop(0)
                            running += i
                            if i == bump:
                                running += 1
                            with self.subTest(x=x, c=c, i=i, running=running):
                                self.assertEqual(x, i)
                                self.assertEqual(c, running)

        def test_ranged(self):
            """Tests specific to RangedHistogram"""
            h = RangedHistogram(xmin=0, xmax=99)
            for i in range(10):
                h.record(i)
            h.record(-100)
            h.record(-200)
            h.record(100)
            self.assertEqual(h.overs, 1)
            self.assertEqual(h.unders, 2)
            self.assertEqual(h.n, 10)

            # unrecording too many out of range
            for testval, ou in ((-100, 'unders'), (100, 'overs')):
                with self.subTest(testval=testval):
                    h = RangedHistogram(xmin=0, xmax=99)
                    with self.assertRaises(ValueError):
                        h.record(testval, n=-1)

                    h = RangedHistogram(xmin=0, xmax=99)
                    h.record(testval, n=2)
                    h.record(testval, n=-1)
                    with self.assertRaises(ValueError):
                        h.record(testval, n=-2)
                    self.assertEqual(getattr(h, ou), 1)

            # half-ranged
            h9 = RangedHistogram(xmin=1)
            h1 = RangedHistogram(xmax=0)
            for i in range(10):
                h9.record(i)
                h1.record(i)
            self.assertEqual(h9.n, 9)
            self.assertEqual(h1.n, 1)

            # badly :) ranged
            hx = RangedHistogram(xmin=100, xmax=0)
            for i in range(10):
                hx.record(i)
            self.assertEqual(hx.n, 0)

        def test_intrange(self):
            """Tests specific to IntRangeHistogram"""
            for nbins in (1, 2, 4, 8):
                h = IntRangeHistogram(nbins=nbins, xmin=0, xmax=255)
                for i in range(256):
                    h.record(i)
                    self.assertTrue(i in h)
                    self.assertFalse(i+10000 in h)
                self.assertEqual(h.n, 256)
                self.assertEqual(len(list(h)), nbins)

            # a test that seems "wrong" but of course is part of the
            # definition/consequence of binning multiple x values together:
            h = IntRangeHistogram(nbins=4, xmin=0, xmax=100)
            h.record(1)
            self.assertTrue(2 in h)

        def test_realrange(self):
            """Tests specific to RealRangeHistogram"""
            nbins = 100
            maxval = float(nbins)
            h = RealRangeHistogram(xmin=0.0, xmax=float(nbins), nbins=nbins)

            # the obvious min/max tests, plus one middle value:
            for x in (0, float(nbins), float(nbins)/2):
                h.clear()
                h.record(x)
                self.assertEqual(h.n, 1)
                self.assertEqual(h[x], 1)

            # more obvious min/max tests
            h.clear()
            h.record(-123456789)
            self.assertEqual(h.n, 0)
            self.assertEqual(h.unders, 1)

            h.record(maxval + 0.0000000000001)
            self.assertEqual(h.n, 0)
            self.assertEqual(h.overs, 1)

            # step through the range by in units of 1 / denom (i.e., a
            # fractional amount), which should result in count of 'denom'
            # in each bin. NOTE: being careful of floating gotchas.
            for denom in (2, 3, 5, 10, 100, 123):
                h.clear()
                for i in range(nbins * denom):
                    h.record(i / float(denom))
                self.assertEqual(h.n, nbins * denom)
                # and each bin should have an equal share
                for i in range(nbins):
                    self.assertEqual(h[i], denom)

        def test_modulus(self):
            """Tests specific to ModuloHistogram"""
            # this counts even/odd
            h = ModuloHistogram(modulus=2)
            for i in range(100):
                h.record(i)
            self.assertEqual(h[0], h[1])
            self.assertEqual(h.n, h[0] + h[1])
            self.assertEqual(h.n, 100)

            # this is how negative numbers work in a Modulo
            modu = 10
            h = ModuloHistogram(modulus=modu)
            negtest = -1
            h.record(negtest)
            self.assertEqual(h[modu + negtest], 1)
            self.assertEqual(h[negtest], 1)

            with self.assertRaises(ValueError):
                h = ModuloHistogram(modulus=0)

            with self.assertRaises(ValueError):
                h = ModuloHistogram(modulus=-1.5)

        def test_binmap(self):
            binmap1 = [
                ([(operator.ge, -1.0), (operator.lt, 0)], -0.5),
                ([(operator.le, 1.0), (operator.ge, 0)], 0.5)
            ]
            h = BinMapHistogram(binmap=binmap1)
            h.record(-1.0)
            h.record(0)
            h.record(1.0)
            self.assertEqual(h.n, 3)
            self.assertEqual(h[-0.5], 1)
            self.assertEqual(h[0.5], 2)

            # this mapping leaves 0.0 unmapped (!) as well as +/- 1.0
            binmap2 = [
                ([(operator.gt, -1.0), (operator.lt, 0)], -0.5),
                ([(operator.lt, 1.0), (operator.gt, 0)], 0.5)
            ]

            h = BinMapHistogram(binmap=binmap2)
            for tv in (0, -1.0, 1.0):
                with self.subTest(tv=tv):
                    with self.assertRaises(ValueError):
                        h.record(tv)
            h.record(-0.9999)
            h.record(0.9999)
            self.assertEqual(h.n, 2)
            self.assertEqual(h[-0.5], 1)
            self.assertEqual(h[0.5], 1)

            # same but with a nomatch function
            h = BinMapHistogram(binmap=binmap2, nomatch_f=lambda x: 17)
            h.record(-1.0)
            h.record(0)
            h.record(1.0)
            self.assertEqual(h.n, 3)
            self.assertEqual(h[0], 3)

            # for grins, verify the mapper is working as expected
            self.assertEqual(h.mapper(123456), 17)

            # somewhat degenerate case... no binmaps JUST the nomatch_f
            h = BinMapHistogram(binmap=[], nomatch_f=lambda x: 17)
            for v in ("banana", 13, None, object()):
                self.assertEqual(h.mapper(v), 17)

            # this default mapper hard-chops down to a multiple of 10
            h = BinMapHistogram(binmap=[], nomatch_f=lambda x: (x//10)*10)
            for i in range(10):
                h.record(i)
            self.assertEqual(h.n, 10)
            self.assertEqual(h[0], h[9])

        def test_dictmap(self):
            ix = 'abcdefghijklmnopqrstuvwxyz'
            m = {c: i for i, c in enumerate(ix)}
            # both of these should result in the same kind of mapping
            # (one is from an iterable, the other an explicit dict)
            hm = DictMapHistogram(dictmap=m)
            hx = DictMapHistogram(dictmap=ix)
            # so test them both
            for h in (hm, hx):
                with self.subTest(h=h):
                    h.record('a')
                    self.assertEqual(h['a'], 1)

                    h.record('a')
                    self.assertEqual(h['a'], 2)

                    with self.assertRaises(KeyError):
                        h.record('!')

                    with self.assertRaises(KeyError):
                        _ = h['@']

            # because it was made with auto-stable, this should work
            h = DictMapHistogram(dictmap=m)
            h.record('q')
            k = list(h)[0]     # i.e., figure out the 1 mapped bin key
            h.record(k)        # and auto-stable should map it to itself
            self.assertEqual(h['q'], 2)    # ... so this should be 2 now

            # same without auto_stable
            h = DictMapHistogram(dictmap=m, auto_stable=False)
            h.record('q')
            k = list(h)[0]
            with self.assertRaises(KeyError):
                h.record(k)
            self.assertEqual(h['q'], 1)

            # test the default mapping feature
            h = DictMapHistogram(dictmap={}, defaultmapped=17)
            teststring = "we all live in a yellow submarine!"
            for c in teststring:
                h.record(c)
            self.assertEqual(h[17], len(teststring))

    unittest.main()
