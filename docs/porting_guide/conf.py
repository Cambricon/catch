# -*- coding: utf-8 -*-
#
# pylint: disable=C0301,C0305
# PyTorch documentation build configuration file, created by
# sphinx-quickstart on Tue Apr 23 18:33:05 2020.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
from __future__ import print_function

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
extensions = [
   'sphinx.ext.autodoc',
   'sphinx.ext.autosummary',
   'sphinx.ext.napoleon',
   'sphinx.ext.doctest',
   'sphinx.ext.mathjax',
   'sphinx.ext.imgmath',
   'breathe'
]

#breathe配置，不需要的可以删除
breathe_projects = { "training": "../doxygen/xml" }
breathe_default_project = "training"

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'寒武纪PyTorch网络移植手册'
copyright = u'2022, Cambricon'
author = u''

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = u'1.1.2'
# The full version, including alpha/beta/rc tags.
release = version
curfnpre=u'Cambricon-PyTorch-Porting-Guide-CN-v'
curfn=curfnpre + version + '.tex'
today = ""

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'zh_CN'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

smartquotes = False
numfig = True
numfig_secnum_depth = 1
# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_permalinks = False
html_copy_source = False
html_css_files = [
    'custom.css',
]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'PyTorchdoc'


# -- Options for LaTeX output ---------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, curfn, u'寒武纪PyTorch网络移植手册',
     author, 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'Cambricon-PyTorch-Porting-Guide', u'寒武纪PyTorch网络移植手册',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'Cambricon-PyTorch-Porting-Guide', u'寒武纪PyTorch网络移植手册',
     author, 'Cambricon', 'One line description of project.',
     'Miscellaneous'),
]


# xelatex 作为 latex 渲染引擎，因为可能有中文渲染
latex_engine = 'xelatex'

 # 如果没有使用自定义封面，可以加上封面 logo。这里可以是 pdf 或者 png，jpeg 等
latex_logo = "./logo.png"


latex_show_urls = 'footnote'

# 主要的配置，用于控制格式等
latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
#
'papersize': 'a4paper',

# The font size ('10pt', '11pt' or '12pt').
#
# 'pointsize': '8pt',

'inputenc': '',
'utf8extra': '',
'figure_align': 'H',
'releasename':"版本",
'sphinxsetup': '''
                verbatimwithframe=false,
                verbatimwrapslines=true,
                VerbatimColor={RGB}{220,220,220},
                verbatimhintsturnover=false,
                ''',
# Additional stuff for the LaTeX preamble.
'preamble': '''
\\addto\\captionsenglish{\\renewcommand{\\chaptername}{}}

% 设置字体

\\usepackage{xeCJK}
\\usepackage{fontspec}
\\setCJKmainfont{Noto Sans CJK SC}
\\setCJKmonofont{Noto Sans CJK SC}
\\setmainfont{Noto Sans CJK SC}

% 段首缩进 2 格

%\\usepackage{indentfirst}
%\\setlength{\\parindent}{2em}

\\usepackage{setspace}

% 1.5 倍行间距
\\renewcommand{\\baselinestretch}{1.5}
% 表格里的行间距
\\renewcommand{\\arraystretch}{1.5}

% list 列表的缩进对齐
\\usepackage{enumitem}
\\setlist{nosep}

% 表格类的宏包
\\usepackage{threeparttable}
\\usepackage{array}
\\usepackage{booktabs}

% fancy 页眉页脚
\\usepackage{fancyhdr}
\\pagestyle{fancy}

% 在 sphinx 生成的 tex 文件里，normal 是指普通页面（每一个章节里，除了第一页外剩下的页面）
% 页眉，L：left，R：right，E：even，O：odd
% 奇数页面：左边是 leftmark，章号和章名称；右边是 rightmark，节号与节名称
% 偶数页面：左边是 rightmark，节号与节名称；右边是 leftmark，章号和章名称
% textsl 是字体，slanted shape，对于英语而言，某种斜体。但是不同于 textit 的 italic shape
% 左页脚：版权信息
% 右页脚：页码数
% rulewidth：页眉和页脚附近的横线的粗细，当设置为 0pt 时，就没有该横线
%
\\fancypagestyle{normal} {
\\fancyhf{}
\\fancyhead{}
\\fancyhead[LE,RO]{\\textsl{\\rightmark}}
\\fancyhead[LO,RE]{\\textsl{\\leftmark}}
\\lfoot{Copyright © 2022 Cambricon Corporation.}
\\rfoot{\\thepage}
\\renewcommand{\\headrulewidth}{0.4pt}
\\renewcommand{\\footrulewidth}{0.4pt}
}

% 在 sphinx 生成的 tex 文件里，plain 是指每个章节的第一页等
\\fancypagestyle{plain} {
\\fancyhf{}
% left head 还可以内嵌图片，图片可以是 pdf，png，jpeg 等
% \\lhead{\\includegraphics[height=40pt]{cn_tm.pdf}}
\\lhead{\\large\\textcolor[rgb]{0.1804,0.4588,0.7137}{Cambricon®}}
\\lfoot{Copyright © 2022 Cambricon Corporation.}
\\rfoot{\\thepage}
\\renewcommand{\\headrulewidth}{0.4pt}
\\renewcommand{\\footrulewidth}{0.4pt}
}
\\tolerance=100000
\\emergencystretch=\\maxdimen
\\hyphenpenalty=10000
\\hbadness=10000
''',


#
'printindex': r'\footnotesize\raggedright\printindex',


# 移除空白页面
'extraclassoptions': 'openany,oneside',

# 如果需要用 latex 自已做封面，可以使用 maketitle
# 下面这个封面的例子来自于互联网

# 'maketitle': r'''
#         \pagenumbering{Roman} %%% to avoid page 1 conflict with actual page 1
#
#         \begin{titlepage}
#             \centering
#
#             \vspace*{40mm} %%% * is used to give space from top
#             \textbf{\Huge {Sphinx format for Latex and HTML}}
#
#             \vspace{0mm}
#             \begin{figure}[!h]
#                 \centering
#                 \includegraphics[width=0.8\textwidth]{cn.png}
#             \end{figure}
#
#             % \vspace{0mm}
#             % \Large \textbf{{Meher Krishna Patel}}
#
#             % \small Created on : Octorber, 2017
#
#             % \vspace*{0mm}
#             % \small  Last updated : \MonthYearFormat\today
#
#
#             %% \vfill adds at the bottom
#             % \vfill
#             % \small \textit{More documents are freely available at }{\href{http://pythondsp.readthedocs.io/en/latest/pythondsp/toc.html}{PythonDSP}}
#         \end{titlepage}
#
#         \clearpage
#         \pagenumbering{roman}
#         \tableofcontents
#         \listoffigures
#         \listoftables
#         \clearpage
#         \pagenumbering{arabic}
#
#         ''',
#
} # latex_elements

import os
import os.path as osp
import sys

sys.path.append(osp.join(osp.dirname(__file__), '.'))

from sphinx.errors import SphinxError
import parsejson
import re
from sphinx.directives.other import TocTree

#自定义cntoctree指令，使其支持only指令
#使用方式：使用toctree指令的地方，直接修改为cntoctree，则在cntoctree指令的下面可以使用only指令，根据条件编译包含的文件。
class TocTreeFilt(TocTree):

    def __returnnewcontent(self,onlystr,contentstr,contentlst):
        #根据only字符串，返回修改后的列表
        #根据正则表达式，解析出only后的tag标签
        regstr = r".. only::([\s\S]*)"
        pattern = re.compile(regstr,re.I|re.U)
        tagstr = pattern.search(onlystr).group(1).strip()
        #print('tagstr=%s' % tagstr)

        if tags.has(tagstr):
            #如果包含该标签，则仅将only指令删除即可
            onlyindex = contentlst.index(onlystr)

            #因为only后面的元素因为需要缩进，因此需要把前后的空格删除
            for i in range(onlyindex,len(contentlst)):
                if i > (onlyindex+1) and (not contentlst[i]):
                    break
                contentlst[i] = contentlst[i].strip()

            contentlst.remove(onlystr)
        else:
            #如果不包含该tag，则将only指令和下一个空元素之间的所有指令都删除
            onlyindex = contentlst.index(onlystr)
            maxindex = len(contentlst)
            #print('onlyindex=%d,maxindex=%d' % (onlyindex,maxindex))

            #循环得到最后一个空元素的索引，以方便从列表里删除
            for i in range(onlyindex,maxindex):
                if (i > (onlyindex+1) and (not contentlst[i])) or (i == maxindex-1):
                    lastindex = i
                    break
            #print('lastindex = %d' % lastindex)
            if lastindex >= maxindex-1:  #如果only在最后
                contentlst = contentlst[0:onlyindex]
            else:
                contentlst = contentlst[0:onlyindex]+contentlst[lastindex:maxindex]
        print(contentlst)
        return contentlst

    def __GetOnlyStrByReg(self,contentstr):
        #根据正则表达式，得到字符串中所有的only完整的字符串，并放到list列表里
        onlylst=[]
        regstr = r"(.. only::[\s\S]*?),"
        pattern = re.compile(regstr,re.I|re.U)
        onlylst = pattern.findall(contentstr)
        return onlylst

    def __filter_only(self,content):
        #解析only指令
        onlystr = '.. only::'
        #将列表转为字符串，用英文逗号链接，方便判断
        contentstr = ','.join(content)

        print(contentstr)
        #判断有没有only指令，有only指令再做解析
        if onlystr in contentstr:
            onlylst=self.__GetOnlyStrByReg(contentstr)
            count = len(onlylst)
            for i in range(0,count):
                content= self.__returnnewcontent(onlylst[i],contentstr,content)

        return content

    def run(self):
        # 过滤only条件
        self.content = self.__filter_only(self.content)
        return super().run()

selffnlst = '' #保存latex_documents数组
warnfile = '' #告警文件，不包含路径
warnfilepath ='' #保存告警日志文件，包含完整的路径名

def config_inited_handler(app,config):
    global selffnlst

    selffnlst = config.latex_documents


def build_finished_handler(app,exception):
    if exception != None:
        print(exception)
        return

    #判断告警文件是否存在，只有无告警或者告警全是忽略告警才允许继续后续的编译
    if warnfilepath !='' and osp.exists(warnfilepath):
        #判断告警文件中是否全是忽略告警
        iswarn = parsejson.warn_main(warnfilepath)
        if iswarn:
            #如果为True则说明有不可忽略的告警，报sphinxerror异常，停止继续编译
            raise SphinxError('There are alarms, please check the file of %s for details' % warnfile)
            return

    parsejson.Modifylatex_main(app.outdir,selffnlst)

def build_inited_handler(app):

    global warnfile
    global warnfilepath

    print(sys.argv)
    args = sys.argv[1:] #0为sphinx-build，需忽略掉
    if '-w' in args:
        pos = args.index('-w') #找到-w所在的索引位置
        warnfile = args[pos+1] #得到告警保存的文件名
        print('warnfile=' + warnfile)

        #根据工作路径，得到文件名的绝对路径
        #当前在build阶段，因此工作路径为Makefile所在的目录，-w后面的文件保存在基于Makefile的相对路径下
        filepath = osp.join(os.getcwd(),warnfile)
        warnfilepath = osp.abspath(filepath)
        print('warnfilepath = ' + warnfilepath)

def doctree_read_handler(app,doctree):
    pass

def doctree_resolved_handler(app, doctree, docname):
    pass

def setup(app):
    app.connect('config-inited', config_inited_handler)
    app.connect('build-finished',build_finished_handler)
    app.connect('builder-inited',build_inited_handler)
    app.connect('doctree-read',doctree_read_handler)
    app.connect('doctree-resolved',doctree_resolved_handler)

    app.add_directive('cntoctree', TocTreeFilt)
    #app.setup_extension('custoctree')

