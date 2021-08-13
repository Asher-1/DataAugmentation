import os
import sys
import shutil
from xml.dom.minidom import Document
from xml.etree.ElementTree import ElementTree, Element
import xml.dom.minidom

JPG = [70, 50, 30]
SCALES = [1.5 ** 0.5, 1.5, 1.5 ** 1.5, 1.5 ** 2, 1.5 ** 2.5]


# 产生变换后的xml文件
def gen_xml(xml_input, trans, outfile):
    for trans in trans.split('*'):
        if trans == "plain" or trans.startswith("jpg") or trans.startswith('color'):  # 如果是这几种变换，直接修改xml文件名就好
            dom = xml.dom.minidom.parse(xml_input)
            root = dom.documentElement
            filenamelist = root.getElementsByTagName('filename')
            filename = filenamelist[0]
            c = str(filename.firstChild.data)
            d = ".".join(outfile.split("\\")[-1].split(".")[:-1]) + '.jpg'
            filename.firstChild.data = d
            f = open(outfile, 'w')
            dom.writexml(f, encoding='utf-8')
        elif trans.startswith("scale"):  # 对于尺度变换，xml文件信息也需要改变
            scale = SCALES[int(trans.replace("scale", ""))]
            dom = xml.dom.minidom.parse(xml_input)
            root = dom.documentElement
            filenamelist = root.getElementsByTagName('filename')
            filename = filenamelist[0]
            c = str(filename.firstChild.data)
            d = ".".join(outfile.split("\\")[-1].split(".")[:-1]) + '.jpg'
            filename.firstChild.data = d
            heightlist = root.getElementsByTagName('height')
            height = heightlist[0]
            a = int(height.firstChild.data)
            b = str(int(a / scale))
            height.firstChild.data = b
            widthlist = root.getElementsByTagName('width')
            width = widthlist[0]
            a = int(width.firstChild.data)
            b = str(int(a / scale))
            width.firstChild.data = b
            objectlist = root.getElementsByTagName('xmin')
            for object in objectlist:
                a = int(object.firstChild.data)
                b = str(int(a / scale))
                object.firstChild.data = b
            objectlist = root.getElementsByTagName('ymin')
            for object in objectlist:
                a = int(object.firstChild.data)
                b = str(int(a / scale))
                object.firstChild.data = b
            objectlist = root.getElementsByTagName('xmax')
            for object in objectlist:
                a = int(object.firstChild.data)
                b = str(int(a / scale))
                object.firstChild.data = b
            objectlist = root.getElementsByTagName('ymax')
            for object in objectlist:
                a = int(object.firstChild.data)
                b = str(int(a / scale))
                object.firstChild.data = b
            f = open(outfile, 'w')
            dom.writexml(f, encoding='utf-8')
        else:
            assert False, "Unrecognized transformation: " + trans


# 产生各种变换名
def get_all_trans():
    transformations = (["plain"]
                       + ["jpg%d" % i for i in JPG]
                       + ["scale0", "scale1", "scale2", "scale3", "scale4"]
                       + ["color%d" % i for i in range(3)])
    return transformations


if __name__ == "__main__":
    inputpath = sys.argv[1]
    name = [name for name in os.listdir(inputpath) if os.path.isfile(os.path.join(inputpath, name))]
    outpath = sys.argv[2]
    if len(sys.argv) >= 4:
        trans = sys.argv[3]
        if not trans.startswith("["):
            trans = [trans]
        else:
            trans = eval(trans)
    else:
        trans = get_all_trans()
    print("Generating transformations and storing in %s" % (outpath))
    for k in name:
        for t in trans:
            xml_input = inputpath + '\\' + k
            gen_xml(xml_input, t, outpath + '\\%s_%s.xml' % (".".join(xml_input.split("\\")[-1].split(".")[:-1]), t))
