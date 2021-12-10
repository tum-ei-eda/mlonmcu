def makeHexArray(fileName):
    out = ""
    with open(fileName, "rb") as f:
        data = f.read(1)
        while data:
            out += "0x" + data.hex() + ", "
            data = f.read(1)
    return out


def convertInOutData(cfg):
    inBufs = []
    outBufs = []
    while cfg.debug and not cfg.ignoreData:
        newBufs = []
        while True:
            inFileName = os.path.join(cfg.modelDir, "input", str(len(inBufs)) + "_" + str(len(newBufs)) + ".bin")
            if not os.path.exists(inFileName):
                inFileName = os.path.join(cfg.modelDir, "input", str(len(inBufs)) + ".bin")
                if os.path.exists(inFileName):
                    newBufs.append(makeHexArray(inFileName))
                break
            newBufs.append(makeHexArray(inFileName))
        inBufs.append(newBufs)
        if len(newBufs) == 0:
            break
        newBufs = []
        while True:
            outFileName = os.path.join(cfg.modelDir, "output", str(len(outBufs)) + "_" + str(len(newBufs)) + ".bin")
            if not os.path.exists(outFileName):
                outFileName = os.path.join(cfg.modelDir, "output", str(len(outBufs)) + ".bin")
                if os.path.exists(outFileName):
                    newBufs.append(makeHexArray(outFileName))
                break
            newBufs.append(makeHexArray(outFileName))
        outBufs.append(newBufs)
        if len(newBufs) == 0:
            raise RuntimeError("Did not find model output for given input")

    out = '#include "ml_interface.h"\n'
    out += "#include <stddef.h>\n"
    out += "const int num_data_buffers_in = " + str(sum([len(buf) for buf in inBufs])) + ";\n"
    out += "const int num_data_buffers_out = " + str(sum([len(buf) for buf in outBufs])) + ";\n"
    for i, buf in enumerate(inBufs):
        for j in range(len(buf)):
            out += "const unsigned char data_buffer_in_" + str(i) + "_" + str(j) + "[] = {" + buf[j] + "};\n"
    for i, buf in enumerate(outBufs):
        for j in range(len(buf)):
            out += "const unsigned char data_buffer_out_" + str(i) + "_" + str(j) + "[] = {" + buf[j] + "};\n"

    var_in = "const unsigned char *const data_buffers_in[] = {"
    var_insz = "const size_t data_size_in[] = {"
    for i, buf in enumerate(inBufs):
        for j in range(len(buf)):
            var_in += "data_buffer_in_" + str(i) + "_" + str(j) + ", "
            var_insz += "sizeof(data_buffer_in_" + str(i) + "_" + str(j) + "), "
    var_out = "const unsigned char *const data_buffers_out[] = {"
    var_outsz = "const size_t data_size_out[] = {"
    for i, buf in enumerate(outBufs):
        for j in range(len(buf)):
            var_out += "data_buffer_out_" + str(i) + "_" + str(j) + ", "
            var_outsz += "sizeof(data_buffer_out_" + str(i) + "_" + str(j) + "), "
    out += var_in + "};\n" + var_out + "};\n" + var_insz + "};\n" + var_outsz + "};\n"

    dataFileName = os.path.join(cfg.cwd, "out", "data.c")
    with open(dataFileName, "w") as f:
        f.write(out)

