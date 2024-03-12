""""
A rough example of a Python implementation of the Nand2Tetris HDL

NAND is a primitive implemented at the hardware level so need to define the logic ourselves
All subsequent gates can be expressed via increasingly complex abstractions of NAND
This is typically provided to the student as a built-in class for Nand2Tetris
All subsequent gates must be implemented by the student
"""

# TODO: gates can be passed function inputs that do nothing
# TODO: test scripts?


class Gate(object):
    def __init__(self):
        # Gates values should only be initialized at runtime
        self.a = None
        self.b = None
        self.c = None
        self.a16 = None
        self.b16 = None
        self.c16 = None
        self.d16 = None
        self.e16 = None
        self.f16 = None
        self.g16 = None
        self.h16 = None
        self.sel = None
        self.sel2 = None
        self.sel3 = None
        self._in = None
        self._in8 = None

    def evaluate(self, a=None, b=None, c=None, _in=None, sel=None, sel2=None, sel3=None, _in8=None, _in16=None,
                 a16=None, b16=None, c16=None, d16=None, e16=None, f16=None, g16=None, h16=None):
        """
        validate input, None = uninitialized
        """
        # TODO: can some of these tests be aggregated?
        if a is not None:
            if type(a) is not str:
                a = bin(a)
            if a not in ("0b0", "0b1"):
                raise RuntimeError("a input must be 1 bit")
            self.a = a

        if b is not None:
            if type(b) is not str:
                b = bin(b)
            if b not in ("0b0", "0b1"):
                raise RuntimeError("b input must be 1 bit")
            self.b = b

        if c is not None:
            if type(c) is not str:
                c = bin(c)
            if c not in ("0b0", "0b1"):
                raise RuntimeError("c input must be 1 bit")
            self.c = c

        if sel is not None:
            if type(sel) is not str:
                sel = bin(sel)
            if sel not in ("0b0", "0b1"):
                raise RuntimeError("sel input must be 1 bit")
            self.sel = sel

        if sel2 is not None:
            if type(sel2) is not str:
                sel2 = bin(sel2)
            if sel2 not in ("0b00", "0b01", "0b10", "0b11"):
                raise RuntimeError("sel2 input must be 2 bits")
            self.sel2 = sel2
        
        if sel3 is not None:
            if type(sel3) is not str:
                sel3 = bin(sel3)
            if int(sel3, 2) < 0 or int(sel3, 2) > 7:
                raise RuntimeError("sel3 input must be 3 bits")
            self.sel3 = sel3
            
        if _in is not None:
            if type(_in) is not str:
                _in = bin(_in)
            if _in not in ("0b0", "0b1"):
                raise RuntimeError("_in input must be 1 bit")
            self._in = _in

        if _in8 is not None:
            if type(_in8) is not str:
                _in8 = bin(_in8)
            if int(_in8, 2) < 0 or int(_in8, 2) > 255:
                raise RuntimeError("_in8 input must be 8 bits")
            self._in8 = _in8
        
        if a16 is not None:
            if type(a16) is not str:
                a16 = bin(a16)
            if int(a16, 2) < 0 or int(a16, 2) > 65535:
                raise RuntimeError("a16 input must be 16 bits")
            self.a16 = a16
        
        if b16 is not None:
            if type(b16) is not str:
                b16 = bin(b16)
            if int(b16, 2) < 0 or int(b16, 2) > 65535:
                raise RuntimeError("b16 input must be 16 bits")
            self.b16 = b16
            
        if c16 is not None:
            if type(c16) is not str:
                c16 = bin(c16)
            if int(c16, 2) < 0 or int(c16, 2) > 65535:
                raise RuntimeError("c16 input must be 16 bits")
            self.c16 = c16
            
        if d16 is not None:
            if type(d16) is not str:
                d16 = bin(d16)
            if int(d16, 2) < 0 or int(d16, 2) > 65535:
                raise RuntimeError("d16 input must be 16 bits")
            self.d16 = d16

        if e16 is not None:
            if type(e16) is not str:
                e16 = bin(e16)
            if int(e16, 2) < 0 or int(e16, 2) > 65535:
                raise RuntimeError("e16 input must be 16 bits")
            self.e16 = e16

        if f16 is not None:
            if type(f16) is not str:
                f16 = bin(f16)
            if int(f16, 2) < 0 or int(f16, 2) > 65535:
                raise RuntimeError("f16 input must be 16 bits")
            self.f16 = f16

        if g16 is not None:
            if type(g16) is not str:
                g16 = bin(g16)
            if int(g16, 2) < 0 or int(g16, 2) > 65535:
                raise RuntimeError("g16 input must be 16 bits")
            self.g16 = g16

        if h16 is not None:
            if type(h16) is not str:
                h16 = bin(h16)
            if int(h16, 2) < 0 or int(h16, 2) > 65535:
                raise RuntimeError("h16 input must be 16 bits")
            self.h16 = h16
            
        # run gate specific logic
        return self.calculate()

    def calculate(self):
        raise NotImplementedError


class NandGate(Gate):
    """
    For two 1 inputs return a 0 output, else return a 1 output
    """
    def calculate(self):
        # cast to int from binary string for comparison
        # endianness does not matter as only 1 bit returned
        if int(self.a, 2) and int(self.b, 2):
            return "0b0"
        else:
            return "0b1"


class NotGate(Gate):
    """
    For a single input, return the opposite
    """
    def calculate(self):
        """
        A single input/output can still have a one-to-many relationship with other gates
        In this case it is passed to both inputs of the NAND gate
        """
        # endianness does not matter as only 1 bit returned
        return NandGate().evaluate(a=self.a, b=self.a)


class Not16Gate(Gate):
    """
    NotGate but with two x 16 bit inputs and one 16 bit output, each bit is compared across both inputs
    """
    def calculate(self):
        byte_str = "0b"
        # will return an LSB endianness result which is then reversed to MSB for Python
        # HACK = LSB = 0000000000000001
        # PYBS = MSB = 1000000000000000
        for i in reversed(range(1, 17)):
            byte_str += NotGate().evaluate(a="0b"+self.a16[i*-1])[2:]
        return byte_str


class AndGate(Gate):
    """
    For two 1 inputs return a 1 output, else return a 0 output
    """
    def calculate(self):
        """
        Now we can combine a NAND with a NOT to get a regular AND gate
        """
        # endianness does not matter as only 1 bit returned
        nand_a = NandGate().evaluate(a=self.a, b=self.b)
        return NotGate().evaluate(a=nand_a)


class And16Gate(Gate):
    """
    AndGate but with two x 16 bit inputs and one 16 bit output, each bit is compared across both inputs
    """
    def calculate(self):
        byte_str = "0b"
        # will return an LSB endianness result which is then reversed to MSB for Python
        # HACK = LSB = 0000000000000001
        # PYBS = MSB = 1000000000000000
        for i in reversed(range(1, 17)):
            byte_str += AndGate().evaluate(a="0b"+self.a16[i*-1], b="0b"+self.b16[i*-1])[2:]
        return byte_str


class OrGate(Gate):
    """
    If either of the two inputs are 1 return a 1 output, else return a 0 output
    """
    def calculate(self):
        # endianness does not matter as only 1 bit returned
        nand_a = NandGate().evaluate(a=self.a, b=self.a)
        nand_b = NandGate().evaluate(a=self.b, b=self.b)
        return NandGate().evaluate(a=nand_a, b=nand_b)


class Or16Gate(Gate):
    """
    OrGate but with two x 16 bit inputs and one 16 bit output, each bit is compared across both inputs
    """
    def calculate(self):
        byte_str = "0b"
        # will return an LSB endianness result which is then reversed to MSB for Python
        # HACK = LSB = 0000000000000001
        # PYBS = MSB = 1000000000000000
        for i in reversed(range(1, 17)):
            byte_str += OrGate().evaluate(a="0b"+self.a16[i*-1], b="0b"+self.b16[i*-1])[2:]
        return byte_str


class OrGate8Way(Gate):
    """
    8 bit bus of 1 bit inputs, 1 bit output, if any bits 1 return 1, else 0
    """
    def calculate(self):
        # endianness does not matter as only 1 bit returned
        or0 = OrGate().evaluate(a="0b"+self._in8[-1], b="0b"+self._in8[-2])
        or1 = OrGate().evaluate(a="0b"+self._in8[-3], b="0b"+self._in8[-4])
        or2 = OrGate().evaluate(a="0b"+self._in8[-5], b="0b"+self._in8[-6])
        or3 = OrGate().evaluate(a="0b"+self._in8[-7], b="0b"+self._in8[-8])

        or4 = OrGate().evaluate(a=or0, b=or1)
        or5 = OrGate().evaluate(a=or2, b=or3)

        return OrGate().evaluate(a=or4, b=or5)


class XorGate(Gate):
    """
    If the two inputs are different return a 1 output, else return a 0 output
    """
    def calculate(self):
        # endianness does not matter as only 1 bit returned
        nand_a = NandGate().evaluate(a=self.a, b=self.b)
        nand_b = NandGate().evaluate(a=self.a, b=nand_a)
        nand_c = NandGate().evaluate(a=nand_a, b=self.b)
        return NandGate().evaluate(a=nand_b, b=nand_c)


class NorGate(Gate):
    """
    If either of the two inputs are 1 return a 0 output, else return a 1 output
    """
    def calculate(self):
        # endianness does not matter as only 1 bit returned
        raise NotImplementedError


class XNorGate(Gate):
    """
    If the two inputs are the same return a 1 output, else return a 0 output
    """
    def calculate(self):
        # endianness does not matter as only 1 bit returned
        raise NotImplementedError


class Mux(Gate):
    """
    Select an output from two inputs, only chosen input will be emitted
    """
    def calculate(self):
        # endianness does not matter as only 1 bit returned
        nand_a = NandGate().evaluate(a=self.sel, b=self.sel)
        nand_b = NandGate().evaluate(a=self.b, b=self.sel)
        nand_c = NandGate().evaluate(a=nand_a, b=self.a)
        return NandGate().evaluate(a=nand_b, b=nand_c)


class Mux16(Gate):
    """
    Mux but with two x 16 bit inputs and one 16 bit output, only chosen input will be emitted
    """
    def calculate(self):
        byte_str = "0b"
        # will return an LSB endianness result which is then reversed to MSB for Python
        # HACK = LSB = 0000000000000001
        # PYBS = MSB = 1000000000000000
        for i in reversed(range(1, 17)):
            byte_str += Mux().evaluate(a="0b"+self.a16[i*-1], b="0b"+self.b16[i*-1], sel=self.sel)[2:]
        return byte_str


class Mux4Way16(Gate):
    """
    Mux16 but with 4 x 16 bit inputs, one 16 bit output, two bit selector, only selected is emitted
    """
    def calculate(self):
        # endianness only matters for selector / result order
        mux16_0 = Mux16().evaluate(a16=self.a16, b16=self.b16, sel="0b"+self.sel2[-1])
        mux16_1 = Mux16().evaluate(a16=self.c16, b16=self.d16, sel="0b"+self.sel2[-1])
        return Mux16().evaluate(a16=mux16_0, b16=mux16_1, sel="0b"+self.sel2[-2])


class Mux8Way16(Gate):
    """
    Mux16 but with 8 x 16 bit inputs, one 16 bit output, 3 bit selector, only selected is emitted
    """
    def calculate(self):
        # endianness only matters for selector / result order
        mux4way16_0 = Mux4Way16().evaluate(a16=self.a16, b16=self.b16, c16=self.c16, d16=self.d16, sel2="0b"+self.sel3[-2]+self.sel3[-1])
        mux4way16_1 = Mux4Way16().evaluate(a16=self.e16, b16=self.f16, c16=self.g16, d16=self.h16, sel2="0b"+self.sel3[-2]+self.sel3[-1])
        return Mux16().evaluate(a16=mux4way16_0, b16=mux4way16_1, sel="0b"+self.sel3[-3])


class DMux(Mux):
    """
    Select one of two outputs, input passes through and unselected output is always 0
    """
    def calculate(self):
        # endianness does not matter as only 2 x 1 bit returned?
        nand_a = NandGate().evaluate(a=self.sel, b=self.sel)
        nand_b = NandGate().evaluate(a=self._in, b=nand_a)
        nand_c = NandGate().evaluate(a=self.sel, b=self._in)
        return NandGate().evaluate(a=nand_b, b=nand_b), NandGate().evaluate(a=nand_c, b=nand_c)


class DMux4Way(DMux):
    """
    With a 2 bit selector choose one of four outputs, input passes through and unselected is always 0
    """
    def calculate(self):
        # endianness only matters for selector / result order
        dmux_0_a, dmux_0_b = DMux().evaluate(_in=self._in, sel="0b"+self.sel2[-2])
        dmux_1_a, dmux_1_b = DMux().evaluate(_in=dmux_0_a, sel="0b"+self.sel2[-1])
        dmux_2_a, dmux_2_b = DMux().evaluate(_in=dmux_0_b, sel="0b"+self.sel2[-1])
        return dmux_1_a, dmux_1_b, dmux_2_a, dmux_2_b


class DMux8Way(DMux):
    """
    With a 3 bit selector choose one of 8 outputs, input passes through and unselected is always 0
    """
    def calculate(self):
        # endianness only matters for selector / result order
        dmux_a, dmux_b = DMux().evaluate(_in=self._in, sel="0b"+self.sel3[-3])
        dmux4_0 = DMux4Way().evaluate(_in=dmux_a, sel2="0b"+self.sel3[-2]+self.sel3[-1])
        dmux4_1 = DMux4Way().evaluate(_in=dmux_b, sel2="0b"+self.sel3[-2]+self.sel3[-1])
        return dmux4_0 + dmux4_1


class HalfAdder(Gate):
    """
    Computes the sum of 2 x 1 bit inputs, output carry bit & sum bit
    """
    def calculate(self):
        carry = AndGate().evaluate(a=self.a, b=self.b)
        _sum = XorGate().evaluate(a=self.a, b=self.b)
        return carry, _sum


class FullAdder(Gate):
    """
    Computes the sum of 3 x 1 bit inputs, output carry bit & sum bit
        IN a, b, c;  // 1-bit inputs
    OUT sum,     // Right bit of a + b + c
        carry;   // Left bit of a + b + c

    PARTS:
    HalfAdder(a=a, b=b, sum=sumAB, carry=carryAB);
    HalfAdder(a=c, b=sumAB, sum=sum, carry=carryABC);
    Or(a=carryABC, b=carryAB, out=carry);

    """
    def calculate(self):
        carry_ab, sum_ab = HalfAdder().evaluate(a=self.a, b=self.b)
        carry_abc, _sum = HalfAdder().evaluate(a=self.c, b=sum_ab)
        carry = OrGate().evaluate(a=carry_abc, b=carry_ab)
        return carry, _sum


def input_unit_test():
    """
    Test input sizes: catch RuntimeException(s)
    """
    # input size unit test
    _gate = Gate()
    for i in range(0, 9):
        try:
            if i == 1:
                test = _gate.evaluate(a="0b10")  # fail
                raise Exception("Unit test failed (continued where it should have excepted)")
            if i == 2:
                test = _gate.evaluate(b="0b10")  # fail
                raise Exception("Unit test failed (continued where it should have excepted)")
            if i == 3:
                test = _gate.evaluate(sel="0b10")  # fail
                raise Exception("Unit test failed (continued where it should have excepted)")
            if i == 4:
                test = _gate.evaluate(_in="0b10")  # fail
                raise Exception("Unit test failed (continued where it should have excepted)")
            if i == 5:
                test = _gate.evaluate(sel2="0b100")  # fail
                raise Exception("Unit test failed (continued where it should have excepted)")
            if i == 6:
                test = _gate.evaluate(_in8="0b0000")  # fail
                raise Exception("Unit test failed (continued where it should have excepted)")
            if i == 7:
                test = _gate.evaluate(a16="0b0000")  # fail
                raise Exception("Unit test failed (continued where it should have excepted)")
            if i == 8:
                test = _gate.evaluate(b16="0b0000")  # fail
                raise Exception("Unit test failed (continued where it should have excepted)")
        except RuntimeError or NotImplementedError:
            continue


def main():
    """
    Sanity check our truth tables for each gate as implemented
    These unit tests are typically provided to the student in Nand2Tetris so they can confirm their results
    """
    _nand = NandGate()
    _not = NotGate()
    _not16 = Not16Gate()
    _and = AndGate()
    _and16 = And16Gate()
    _or = OrGate()
    _or16 = Or16Gate()
    _or8way = OrGate8Way()
    _xor = XorGate()
    # TODO: nor, xnor
    _mux = Mux()
    _mux16 = Mux16()
    _mux4way16 = Mux4Way16()
    _mux8way16 = Mux8Way16()
    _dmux = DMux()
    _dmux4way = DMux4Way()
    _dmux8way = DMux8Way()
    _halfAdder = HalfAdder()
    _fullAdder = FullAdder()

    input_unit_test()

    # For two 1 inputs return a 1 output, else return a 1 output
    assert _nand.evaluate(a="0b1", b="0b1") == "0b0"
    assert _nand.evaluate(a="0b1", b="0b0") == "0b1"
    assert _nand.evaluate(a="0b0", b="0b1") == "0b1"
    assert _nand.evaluate(a="0b0", b="0b0") == "0b1"

    # For a single input, return the opposite
    assert _not.evaluate(a="0b1") == "0b0"
    assert _not.evaluate(a="0b0") == "0b1"

    # NotGate but with two x 16 bit inputs and one 16 bit output, each bit is compared across both inputs
    assert _not16.evaluate(a16="0b0000000000000000") == "0b1111111111111111"
    assert _not16.evaluate(a16="0b1111111111111111") == "0b0000000000000000"
    assert _not16.evaluate(a16="0b0000001111000000") == "0b1111110000111111"

    # For two 1 inputs return a 1 output, else return a 0 output
    assert _and.evaluate(a="0b1", b="0b1") == "0b1"
    assert _and.evaluate(a="0b1", b="0b0") == "0b0"
    assert _and.evaluate(a="0b0", b="0b1") == "0b0"
    assert _and.evaluate(a="0b0", b="0b0") == "0b0"

    # AndGate but with two x 16 bit inputs and one 16 bit output, each bit is compared across both inputs
    assert _and16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000") == "0b0000000000000000"
    assert _and16.evaluate(a16="0b1111111111111111", b16="0b1111111111111111") == "0b1111111111111111"
    assert _and16.evaluate(a16="0b0000001111000000", b16="0b0000000000000000") == "0b0000000000000000"
    assert _and16.evaluate(a16="0b0000001111000000", b16="0b0000001111000000") == "0b0000001111000000"

    # If either of the two inputs are 1 return a 1 output, else return a 0 output
    assert _or.evaluate(a="0b1", b="0b1") == "0b1"
    assert _or.evaluate(a="0b1", b="0b0") == "0b1"
    assert _or.evaluate(a="0b0", b="0b1") == "0b1"
    assert _or.evaluate(a="0b0", b="0b0") == "0b0"

    # OrGate but with two x 16 bit inputs and one 16 bit output, each bit is compared across both inputs
    assert _or16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000") == "0b0000000000000000"
    assert _or16.evaluate(a16="0b1111111111111111", b16="0b1111111111111111") == "0b1111111111111111"
    assert _or16.evaluate(a16="0b0000001111000000", b16="0b0000000000000000") == "0b0000001111000000"
    assert _or16.evaluate(a16="0b1111000000000000", b16="0b0000000000000000") == "0b1111000000000000"

    # 8 bit bus of 1 bit inputs, 1 bit output, if any bits 1 return 1, else 0
    assert _or8way.evaluate(_in8="0b11111111") == "0b1"
    assert _or8way.evaluate(_in8="0b00011000") == "0b1"
    assert _or8way.evaluate(_in8="0b00000000") == "0b0"

    # If the two inputs are different return a 1 output, else return a 0 output
    assert _xor.evaluate(a="0b1", b="0b1") == "0b0"
    assert _xor.evaluate(a="0b1", b="0b0") == "0b1"
    assert _xor.evaluate(a="0b0", b="0b1") == "0b1"
    assert _xor.evaluate(a="0b0", b="0b0") == "0b0"

    # Select an output from two inputs, only chosen input will be emitted
    assert _mux.evaluate(a="0b1", b="0b0", sel="0b0") == "0b1"
    assert _mux.evaluate(a="0b1", b="0b0", sel="0b1") == "0b0"
    assert _mux.evaluate(a="0b0", b="0b1", sel="0b0") == "0b0"
    assert _mux.evaluate(a="0b0", b="0b1", sel="0b1") == "0b1"

    # Mux but with two x 16 bit inputs and one 16 bit output, each bit is compared across both inputs
    assert _mux16.evaluate(a16="0b1111111111111111", b16="0b0000000000000000", sel="0b0") == "0b1111111111111111"
    assert _mux16.evaluate(a16="0b1111111111111111", b16="0b0000000000000000", sel="0b1") == "0b0000000000000000"
    assert _mux16.evaluate(a16="0b0000000000000000", b16="0b1111111111111111", sel="0b0") == "0b0000000000000000"
    assert _mux16.evaluate(a16="0b0000000000000000", b16="0b1111111111111111", sel="0b1") == "0b1111111111111111"

    # Mux16 but with 4 x 16 bit inputs, one 16 bit output, two bit selector, only selected is emitted
    assert _mux4way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", sel2="0b00") == "0b0000000000000000"
    assert _mux4way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", sel2="0b01") == "0b0000000000000000"
    assert _mux4way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", sel2="0b10") == "0b0000000000000000"
    assert _mux4way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", sel2="0b11") == "0b0000000000000000"
    assert _mux4way16.evaluate(a16="0b1111111111111111", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", sel2="0b00") == "0b1111111111111111"
    assert _mux4way16.evaluate(a16="0b0000000000000000", b16="0b1111111111111111", c16="0b0000000000000000", d16="0b0000000000000000", sel2="0b01") == "0b1111111111111111"
    assert _mux4way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b1111111111111111", d16="0b0000000000000000", sel2="0b10") == "0b1111111111111111"
    assert _mux4way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b1111111111111111", sel2="0b11") == "0b1111111111111111"

    # Mux16 but with 8 x 16 bit inputs, one 16 bit output, 3 bit selector, only selected is emitted
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b000") == "0b0000000000000000"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b001") == "0b0000000000000000"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b010") == "0b0000000000000000"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b011") == "0b0000000000000000"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b100") == "0b0000000000000000"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b101") == "0b0000000000000000"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b110") == "0b0000000000000000"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b111") == "0b0000000000000000"
    assert _mux8way16.evaluate(a16="0b1111111111111111", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b000") == "0b1111111111111111"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b1111111111111111", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b001") == "0b1111111111111111"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b1111111111111111", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b010") == "0b1111111111111111"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b1111111111111111", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b011") == "0b1111111111111111"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b1111111111111111", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b100") == "0b1111111111111111"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b1111111111111111", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b101") == "0b1111111111111111"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b1111111111111111", h16="0b0000000000000000", sel3="0b110") == "0b1111111111111111"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b1111111111111111", sel3="0b111") == "0b1111111111111111"

    # Select one of two outputs, input passes through and unselected output is always 0
    assert _dmux.evaluate(_in="0b0", sel="0b0") == ("0b0", "0b0")
    assert _dmux.evaluate(_in="0b0", sel="0b1") == ("0b0", "0b0")
    assert _dmux.evaluate(_in="0b1", sel="0b0") == ("0b1", "0b0")
    assert _dmux.evaluate(_in="0b1", sel="0b1") == ("0b0", "0b1")

    # With a two bit selector choose one of four outputs, input passes through and unselected is always 0
    assert _dmux4way.evaluate(_in="0b0", sel2="0b00") == ("0b0", "0b0", "0b0", "0b0")
    assert _dmux4way.evaluate(_in="0b0", sel2="0b01") == ("0b0", "0b0", "0b0", "0b0")
    assert _dmux4way.evaluate(_in="0b0", sel2="0b10") == ("0b0", "0b0", "0b0", "0b0")
    assert _dmux4way.evaluate(_in="0b0", sel2="0b11") == ("0b0", "0b0", "0b0", "0b0")
    assert _dmux4way.evaluate(_in="0b1", sel2="0b00") == ("0b1", "0b0", "0b0", "0b0")
    assert _dmux4way.evaluate(_in="0b1", sel2="0b01") == ("0b0", "0b1", "0b0", "0b0")
    assert _dmux4way.evaluate(_in="0b1", sel2="0b10") == ("0b0", "0b0", "0b1", "0b0")
    assert _dmux4way.evaluate(_in="0b1", sel2="0b11") == ("0b0", "0b0", "0b0", "0b1")
    
    # With a 3 bit selector choose one of 8 outputs, input passes through and unselected is always 0
    assert _dmux8way.evaluate(_in="0b0", sel3="0b000") == ("0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b0", sel3="0b001") == ("0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b0", sel3="0b010") == ("0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b0", sel3="0b011") == ("0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b0", sel3="0b100") == ("0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b0", sel3="0b101") == ("0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b0", sel3="0b110") == ("0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b0", sel3="0b111") == ("0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b1", sel3="0b000") == ("0b1", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b1", sel3="0b001") == ("0b0", "0b1", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b1", sel3="0b010") == ("0b0", "0b0", "0b1", "0b0", "0b0", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b1", sel3="0b011") == ("0b0", "0b0", "0b0", "0b1", "0b0", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b1", sel3="0b100") == ("0b0", "0b0", "0b0", "0b0", "0b1", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b1", sel3="0b101") == ("0b0", "0b0", "0b0", "0b0", "0b0", "0b1", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b1", sel3="0b110") == ("0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b1", "0b0")
    assert _dmux8way.evaluate(_in="0b1", sel3="0b111") == ("0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b1")

    # Computes the sum of 2 x 1 bit inputs, output carry bit & sum bit
    assert _halfAdder.evaluate(a="0b0", b="0b0") == ("0b0", "0b0")
    assert _halfAdder.evaluate(a="0b0", b="0b1") == ("0b0", "0b1")
    assert _halfAdder.evaluate(a="0b1", b="0b0") == ("0b0", "0b1")
    assert _halfAdder.evaluate(a="0b1", b="0b1") == ("0b1", "0b0")

    # Computes the sum of 3 x 1 bit inputs, output carry bit & sum bit
    assert _fullAdder.evaluate(a="0b0", b="0b0", c="0b0") == ("0b0", "0b0")
    assert _fullAdder.evaluate(a="0b1", b="0b1", c="0b1") == ("0b1", "0b1")
    assert _fullAdder.evaluate(a="0b1", b="0b0", c="0b0") == ("0b0", "0b1")
    assert _fullAdder.evaluate(a="0b0", b="0b1", c="0b0") == ("0b0", "0b1")
    assert _fullAdder.evaluate(a="0b0", b="0b0", c="0b1") == ("0b0", "0b1")
    assert _fullAdder.evaluate(a="0b0", b="0b1", c="0b1") == ("0b1", "0b0")
    assert _fullAdder.evaluate(a="0b1", b="0b0", c="0b1") == ("0b1", "0b0")
    assert _fullAdder.evaluate(a="0b1", b="0b1", c="0b0") == ("0b1", "0b0")


if __name__ == "__main__":
    main()
