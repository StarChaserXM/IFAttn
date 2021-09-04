import os
import tempfile
import time
import shutil
from pathlib import Path
from subprocess import run, PIPE

from module.utils import system, get_file_type, do_multiprocess

import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO, logger=logger)


class IDAScript:
    def __init__(
        self,
        idapath="/XXX/ida-7.5",
        idc=None,
        idcargs="",
        force=False,
        log=False,
        stdout=False,
        debug=False,
    ):
        self.idapath = idapath
        self.idc = idc
        self.idcargs = idcargs
        self.force = force
        self.log = log
        self.stdout = stdout
        self.debug = debug

        self.env = os.environ.copy()
        self.env["TERM"] = "xterm"
        self.env["TVHEADLESS"] = "1"
        if self.debug:
            coloredlogs.install(level=logging.DEBUG, logger=logger)
            self.force = True
            self.stdout = True

    def remove_leftovers(self, input_fname):
        exts = [".id0", ".id1", ".nam", ".til", ".id2"]
        basename = os.path.splitext(input_fname)[0]
        for ext in exts:
            try:
                os.remove(basename + ext)
            except:
                pass

    def is_done(self, input_fname):
        return os.path.exists(input_fname + ".done")

    def handle_log(self, input_fname, tmp_fname):
        if not os.path.exists(tmp_fname):
            return

        if self.stdout:
            with open(tmp_fname, "rb") as f:
                data = f.read()
            print(data.decode())

        if self.log:
            res_fname = input_fname + ".output"
            shutil.move(tmp_fname, res_fname)
        else:
            os.unlink(tmp_fname)

    def run_helper(self, input_fname):
        if not os.path.exists(input_fname):
            return input_fname, None

        arch = get_file_type(input_fname)
        print(arch)
        if arch is None:
            logger.warn("Skip Unknown file type: %s" % input_fname)
            return input_fname, False

        if not self.force and self.is_done(input_fname):
            return input_fname, True

        self.remove_leftovers(input_fname)

        idc_args = [self.idc]
        idc_args.extend(self.idcargs)
        idc_args = " ".join(idc_args)

        if arch.find("_32") != -1:
            ida = 'wine ' + self.idapath + "/idal.exe"
        else:
            ida = 'wine ' + self.idapath + "/idal64.exe"

        # >= IDA Pro v7.4 use "idat" instead of "idal"
        if not os.path.exists(ida):
            ida = ida.replace('idal.exe', 'idat.exe')

        # Setup command line arguments
        path = [ida, "-A", "-S{}".format(idc_args)]
        if self.log or self.stdout:
            fd, tmp_fname = tempfile.mkstemp()
            os.close(fd)
            # IDA supports logging by '-L'
            path.append("-L{}".format(tmp_fname))
        path.append(input_fname)
        logger.debug(" ".join(path))

        ret = run(path, env=self.env, stdout=PIPE).returncode
        if self.log or self.stdout:
            self.handle_log(input_fname, tmp_fname)
        if ret != 0:
            logger.error("IDA returned {} for {}".format(ret, input_fname))
            return input_fname, False
        else:
            Path(input_fname + ".done").touch()
            return input_fname, True

    def get_elf_files(self, input_path):
        if os.path.isdir(input_path):
            cmd = "find {0} -type f -executable | grep ELF".format(input_path)
            fnames = system(cmd)
        else:
            with open(input_path, "r") as f:
                fnames = f.read().splitlines()
        return fnames

    def run(self, input_path):
        elfs = self.get_elf_files(input_path)

        logger.info("[+] start extracting {0} files ...".format(len(elfs)))
        t0 = time.time()

        if self.debug:
            elfs = [elfs[0]]
            print(elfs)
        res = do_multiprocess(self.run_helper, elfs, chunk_size=1, threshold=1)
        logger.info("done in: (%0.3fs)" % (time.time() - t0))
        return res
