#!/usr/bin/python
# -*- coding: UTF-8

import os
import sys
import json
import re
import codecs
import shutil
import platform
#import source.conf

Py_version = sys.version_info
Py_v_info = str(Py_version.major) + '.' + str(Py_version.minor) + '.' + str(Py_version.micro)

if Py_version >= (3,5,0) and Py_version <(3,7,0):
    print(False)
    
if Py_version >= (3,7,0):
    print(True)
    
if platform.system().lower() == 'windows':
    print("windows")
elif platform.system().lower() == 'linux':
    print("linux")


def __openconfjsonfile__():
        with codecs.open("conf.json","r+",encoding = 'utf-8') as load_f:
            load_dict = json.load(load_f)
            return load_dict


def GetOtherIncludePackage():
#    packagearr = __load_dict__['package']
#    for package in packagearr:
#        print(package)
    return __load_dict__['package']

def GetReplacePackage():
    return __load_dict__['replacepackage']

def GetCustomOptions():
    return __load_dict__['customoptions']

def GetIsTocContents():
    return __load_dict__['isfiguretabletoc']

def GetSensitiveword():
    #得到敏感词数组，便于搜索文档中是否有敏感词存在
    return __load_dict__['sensitivewords']

def GetTablesContent():
    return __load_dict__['tables']

def GetTablerowtype():
#    packagearr = __load_dict__['tables']['rowtype']
#    print(packagearr)
    return __load_dict__['tables']['rowtype']

def GetTableheadtype():
#    packagearr = __load_dict__['tables']['headtype']
#    print(packagearr)
    return __load_dict__['tables']['headtype']

def GetTableHeadFontColor():
    return __load_dict__['tables']['headfontcolor']

def GetTableStylesArr():
#    packagearr = __load_dict__['tables']['styles']
#    for package in packagearr:
#        print(package)
    return __load_dict__['tables']['styles']

def GetImageStyleArr():
#    packagearr = __load_dict__['image']['styles']
#    for package in packagearr:
#        print(package)
    return __load_dict__['image']['styles']


class clsTableattr:

    def __init__(self, tables):
        self.rowtype = tables['rowtype']
        self.headtype = tables['headtype']
        self.headfontcolor = tables['headfontcolor']
        self.tablestyles = tables['styles']

class clsModifyTex:

    def __init__(self, content):
        self.content = content
        self.tablesattrobj = clsTableattr(GetTablesContent())

    #加入其它包
    def AddPackageToTex(self):
        #得到需要包的数组
        packarr = GetOtherIncludePackage()
        if len(packarr)==0:
            return  False;

        #如果数组有内容，就需要将包添加到latex文件的导言区
        #搜索\usepackage{sphinx}，将包加在它的前面，用正则表达式搜索它的位置
        #采用正则表达式替换的方式，替换搜索到的字符串，因此需要先构建字符串
        #python认为\u后面为unicode字符，因此需要多加一个转义字符\，python才认为是整个的字符串
        #searchstr = r'\\usepackage\[dontkeepoldnames\]{sphinx}'
        searchstr = r'\\usepackage(\[\S*\]*)?{sphinx}'
        matchstr = re.search(searchstr,self.content)
        
        replacestr=""
        for package in packarr:
            replacestr  += package+'\n'
             
        if Py_version >= (3,7,0):
            replacestr += "\\" + matchstr.group(0)
        else:
            replacestr += matchstr.group(0)
        
       
        self.content = re.sub(searchstr, replacestr, self.content, 1, re.M | re.I|re.U)

        return True

    #加入自定义选项,包放在了sphinx包的前面，因此选项放在sphinx包的后面
    def AddCustormOptionsToTex(self):
        #得到需要包的数组
        packarr = GetCustomOptions()
        if len(packarr)==0:
            return  False;

        #如果数组有内容，就需要将包添加到latex文件的导言区
        #搜索\usepackage{sphinx}，将自定义参数放在它的后面，用正则表达式搜索它的位置
        #采用正则表达式替换的方式，替换搜索到的字符串，因此需要先构建字符串
        #python认为\u后面为unicode字符，因此需要多加一个转义字符\，python才认为是整个的字符串
        searchstr = r'\\usepackage(\[\S*\]*)?{sphinx}'
        matchstr = re.search(searchstr,self.content)

        replacestr=""
        for package in packarr:
             replacestr  += package+'\n'
             
        if Py_version >= (3,7,0):
            replacestr = "\\" + matchstr.group(0)+'\n'+replacestr
        else:
            replacestr = matchstr.group(0)+'\n'+replacestr

        self.content = re.sub(searchstr, replacestr, self.content, 1, re.M | re.I|re.U)
        return True
     
    #增加figure和table toc到tex
    def AddOtherTocToTex(self):
        #得到需要包的数组
        packarr = GetIsTocContents()
        if len(packarr)==0:
            return  False;

        replacestr = ""
        if packarr['isfigurestoc']:
           figlst = packarr['figurestoc']
           for figstr in figlst:
               replacestr += figstr + '\n'

        if packarr['istablestoc']:
           figlst = packarr['tablestoc']
           for figstr in figlst:
               replacestr += figstr + '\n'
        if replacestr == "":
           return

        #如果数组有内容，就需要将包添加到latex文件的导言区
        #搜索\usepackage{sphinx}，将包加在它的前面，用正则表达式搜索它的位置
        #采用正则表达式替换的方式，替换搜索到的字符串，因此需要先构建字符串
        #python认为\u后面为unicode字符，因此需要多加一个转义字符\，python才认为是整个的字符串
        searchstr = r'\\sphinxtableofcontents'
        matchstr = re.search(searchstr,self.content)

        if Py_version >= (3,7,0):
            replacestr = "\\" + matchstr.group(0) + '\n' + replacestr
        else:
            replacestr = matchstr.group(0) + '\n' + replacestr

        self.content = re.sub(searchstr, replacestr, self.content, 1, re.M | re.I|re.U)
        return True


    #得到需要替换的包，用正则表达式替换
    def ModifyReplacePackage(self):
        #得到字典值
        redict = GetReplacePackage()
        if len(redict) ==0:
           return;
        
        #返回字典中所有键值的列表
        keylst = list(redict)
        for key in keylst:
            if key == 'comment' :
                continue;
            keyvalue = redict[key]   #得到键对应的值
            
            #对键值进行替换
            self.content = re.sub(key, keyvalue, self.content, 0, re.M | re.I|re.U)
        return;

    def ModifyTablesAttributes(self):
        #修改表格属性
        newcontent = self.content
        searchstr = r'(\\begin{savenotes}\\sphinxattablestart|\\begin{savenotes}\\sphinxatlongtablestart)([\s\S]*?)(\\sphinxattableend\\end{savenotes}|\\sphinxatlongtableend\\end{savenotes})'
        m = re.finditer(searchstr, self.content, re.M|re.I|re.U)
        for match in m:
            oldtablestr = match.group()
            tablestr = match.groups()
            caption_dict = self.__CreateTableCaptionDict(self.tablesattrobj.tablestyles)
            if len(caption_dict ) > 0 :
                newtableattr = self.__ModifySingleTableattr(tablestr[0]+tablestr[1]+tablestr[2],caption_dict ) #tablestr也是3个内容的数组，因为正则表达式被分为了3组，只取中间分组的内容。
                #重新组成新的字符串
                newcontent = newcontent.replace(tablestr[0]+tablestr[1]+tablestr[2], newtableattr)
                
        newcontent = newcontent.replace("\\begin{savenotes}\\sphinxatlongtablestart", "\\begin{footnotesize}\\begin{savenotes}\\sphinxatlongtablestart")
        newcontent = newcontent.replace("\\sphinxatlongtableend\\end{savenotes}", "\\sphinxatlongtableend\\end{savenotes}\\end{footnotesize}")
        self.content = newcontent


    def __CreateTableCaptionDict(self, tablestylesarr):
        #根据caption生成表格字典，key=caption，value=属性数组
        cap_dict = {}

        for tablestyle_dict in tablestylesarr:
            captionarr = tablestyle_dict['caption']
            #该caption可能是一个逗号分隔的字符串数组，因此需要做拆分
            captionlist = captionarr.split(",")
            for caption in captionlist:
                cap_dict[caption] = tablestyle_dict  #以caption为key重新生成字典，便于查找
        return cap_dict

    def __ModifySingleTableattr(self, singletablecontent, caption_dict):
        #修改单个表格属性
        #从单个表格里用正则表达式找caption
        #定义正则表达式,查找caption内容
        new_singletablecontent = singletablecontent
        searchstr = r'(\\sphinxcaption|\\caption){(?P<caption>[\s\S]*?)}'
        matchcaption = re.search(searchstr, singletablecontent, re.M | re.I|re.U)
        if matchcaption != None:
            tablecaption = matchcaption.group('caption') #得到caption的值
        else:
            tablecaption = ''
        if tablecaption in caption_dict:
            #修改表格属性
            tablestyle_dict = caption_dict[tablecaption]

                #修改表格为长表格
            new_singletablecontent = self.__StartModifyTableAttr(singletablecontent, 
                                                                 tablestyle_dict['isLongTable'],
                                                                 tablestyle_dict['isCusHead'])
            #渲染竖型表格的第一列
            if tablestyle_dict['isVertical']==True:
                new_singletablecontent = self.__ModifyVerticalTable(new_singletablecontent)
        else:
            #修改表格的通用属性
            new_singletablecontent = self.__StartModifyTableAttr(singletablecontent, False)
        if new_singletablecontent == '':
           new_singletablecontent = singletablecontent

        return new_singletablecontent

    def __StartModifyTableAttr(self, singletablecontent, islongtable,isCusHead=True):
        #修改表格属性
        searchstr = r'(\\begin{tabular}|\\begin{tabulary})(\[[a-z]\]|{\\linewidth}\[[a-z]\])([\s\S].*)'
        #为了添加表格的通用属性，先对字符串做分割
        #python会把正则表达式中的分组自动分割，因此上面的正则表达式会自动分割为三个字符串
        #加上头尾字符串总共分为5个字符串数组。要修改第1维字符串为\\being{longtable},第2维字符串直接删除，第3维字符串不变
        splittable = re.split(searchstr, singletablecontent,0, re.M | re.I|re.U )
        if splittable == None or len(splittable) < 5:
           	 #再修改长表格属性
            searchstr = r'\\begin{longtable}([\s\S].*)'
            #为了添加表格的通用属性，先对字符串做分割
            #python会把正则表达式中的分组自动分割，因此上面的正则表达式会自动分割为三个字符串
            #加上头尾字符串总共分为5个字符串数组。要修改第1维字符串为\\being{longtable},第2维字符串直接删除，第3维字符串不变
            splittable = re.split(searchstr, singletablecontent,0, re.M | re.I|re.U )
            if len(splittable) < 3 or isCusHead==False:
                #至少是3维的数组，否则不是预想的内容，不做处理
                return singletablecontent
            newtable4 = self.__ModifyLongTableHead(splittable[2], self.tablesattrobj.headtype)
            singletablecontent = splittable[0]+r'\begin{longtable}'+splittable[1]+newtable4  #begin{longtable}必须再加上，因为Python并不认为它是正则表达式，因此不再分组里面第0个分组为空。
            return singletablecontent

        #拆分后splittable应该为5个字符串的数组，拆分后便于添加通用属性
        if self.tablesattrobj.rowtype != '':
            splittable[0] += self.tablesattrobj.rowtype + '\n'

        if isCusHead == True:
            #修改表头字体颜色为白色加粗
            newtable4 = self.__ModifyTableHead(splittable[4], self.tablesattrobj.headtype)
        else:
            newtable4 = splittable[4]
        
        singletablecontent = splittable[0]+splittable[1]+splittable[2]+splittable[3]+newtable4
        if islongtable: #如果为长表格要做长表格的替换
            singletablecontent = self.__ModifyTableLongHeadAndTail(singletablecontent)

        return singletablecontent
    
    def __ModifyTableLongHeadAndTail(self,singletablecontent):
        #长表格的头尾都要替换，因此单独封装成一个函数，否则长表格的表格需号将加2
        #替换第一行为长表格
        searchstr = r'(\\begin{savenotes}\\sphinxattablestart)'
        splittable = re.search(searchstr, singletablecontent,re.M | re.I|re.U )
        #替换为长表格
        tablefirstline = re.sub(r'\\sphinxattablestart',r'\\sphinxatlongtablestart', splittable.group(0), re.M|re.I|re.U)
        #替换第2行为长表格
        searchstr = r'(\\begin{tabular}|\\begin{tabulary})(\[[a-z]\]|{\\linewidth}\[[a-z]\])([\s\S].*)'
        splittable = re.search(searchstr, singletablecontent,re.I|re.U )
        #记录表头的位置
        headlastpos = splittable.end()
        tablesecondline = re.sub(r'\\begin{tabular}|\\begin{tabulary}',r'\\begin{longtable}', splittable.group(0), re.I|re.U)
        tablesecondline = re.sub(r'\{\\linewidth\}',r'', tablesecondline, re.I|re.U)

        #查找caption
        searchstr = r'\\sphinxcaption([\s\S].*)'
        splittable = re.search(searchstr, singletablecontent, re.I|re.U)
        longcaption = re.sub(r'\\sphinxcaption',r'\\caption', splittable.group(0), re.I|re.U)
        #添加长表格专用指令
        longcaption += r"\\*[\sphinxlongtablecapskipadjust]"
        
        #构建长表个的表头部分
        longhead = tablefirstline + tablesecondline + '\n' + r'\sphinxthelongtablecaptionisattop' + '\n'+ longcaption+'\n'
        
        #替换表尾
        newtablecontent = singletablecontent[headlastpos:len(singletablecontent)]
        endtable = re.sub(r'(\\end{tabular}|\\end{tabulary})',r'\\end{longtable}', newtablecontent, re.M | re.I|re.U)
        endtable = re.sub(r'\\par',r'', endtable, re.M | re.I|re.U)
        endtable = re.sub(r'\\sphinxattableend',r'\\sphinxatlongtableend', endtable, re.M | re.I|re.U)
        
        #生成新的长表格内容
        singletablecontent = longhead + endtable
        return singletablecontent
        
    #为表格增加h属性对齐方式，避免表格浮动，让表格在当前位置显示    
    def __AddHAttrFroTable(self,content):
    
        searchstr = r'(\\begin{tabular}|\\begin{tabulary}|\\begin{longtable})({\\linewidth})?(\[(?P<attr>[a-z]{1,4})\])?([\s\S].*)'
        m = re.search(searchstr, content, re.M|re.I|re.U )
        attrcontent = m.group('attr')
        posarr = m.span()
        if m.group(3) == '':   #group(3)是[htp]的组合，如果没有表格属性则添加上表格属性
            newcontent = content[0:posarr[0]]+m.group(1)+m.group(2)+r'[h]'+m.group(5)+content[posarr[1]:len(content)]
        else:
            replacestr = m.group(4)  #需要被替换的表格属性
            #判断该表格是否有P属性，如果有p属性需要删掉
            replacestr.replace('p','') 
            #给该表格添加h属性，避免表格浮动，让表格在当前位置显示
            replacestr = 'h'+replacestr
            newcontent = content[0:posarr[0]]+m.group(1)+m.group(2)+'['+replacestr+']'+m.group(5)+content[posarr[1]:len(content)]
        #print('newcontent: %s' % newcontent)
        return newcontent
        
    #修改sphinx自动生成的长表格表头
    def __ModifyLongTableHead(self,content,headtype):

        #先找出第一行
        searchstr = r'\\hline(?P<content>[\s\S]*?)\\hline'
        pattern = re.compile(searchstr,re.M | re.I|re.U)
        matchiter = pattern.finditer(content)
        posarr = []
        i = 0
        for m in matchiter:
            
            if i > 1:
                break;
                
            posarr.append([])
            posarr[i] = m.span() #保存起始位置和结束位置，便于组成新的内容
            
            if i ==0:
                newcontent = content[0:posarr[i][0]]
            else:
                newcontent = newcontent+content[posarr[i-1][1]:posarr[i][0]]
                
            newcontent += r'\hline\rowcolor'+headtype
            
            headcontent = m.group(1) #匹配到的第一个即为表头内容
            
            if 'multicolumn' in headcontent:
                return content        
            
            headlist = []
            
            if r'\sphinxstyletheadfamily' in headcontent:
                pattern = re.compile(r'(?<=\\sphinxstyletheadfamily)(?P<value>[\s\S]*?)(?=(\\unskip|&)|\\\\)', re.M | re.I|re.U)
                aftercontent = headcontent
                mobjarr = pattern.finditer(aftercontent)
                
                preposlist = []
                for mobj in mobjarr:
                    amarr = mobj.group('value')
                    curposlist = mobj.span()
            
                    #用表头内容数组替换
                    fontcolor = self.tablesattrobj.headfontcolor
                    #先去掉首尾空格，避免首尾有空格无法去掉回车换行符
                    amarr = amarr.strip()
                    amarr = amarr.strip('\r')
                    amarr = amarr.strip('\n')
                    amarr = amarr.strip()  #再去掉首尾空格，避免多余的空格出现
                    if amarr == '':
                        continue
                    fontcolor = fontcolor.replace('{}','{'+ amarr+'}',1)
                    if len(preposlist) > 0:
                        headlist.append(headcontent[preposlist[1]:curposlist[0]])
                    else:
                        headlist.append(headcontent[0:curposlist[0]])
                    headlist.append(fontcolor)
                    preposlist = curposlist
                headlist.append(headcontent[preposlist[1]:len(headcontent)])  #把最后一个字符串加上
                headcontent = ''
                for prelist in headlist:
                    headcontent = headcontent + prelist + '\n'
                newcontent += headcontent+r'\hline'               
            i +=1
        newcontent += content[posarr[i-1][1]:len(content)]
        return newcontent

    def __ModifyTableHead(self, content, headtype):
        #先找出第一行
        searchstr = r'\\hline(?P<content>[\s\S]*?)\\hline'
        m = re.search(searchstr, content, re.M|re.I|re.U )
        headcontent = m.group(1) #匹配到的第一个即为表头内容
        posarr = m.span(1)  #保存起始位置和结束位置，便于组成新的内容

        if 'multicolumn' in headcontent:
            return content        

        if r'\sphinxstyletheadfamily' in headcontent:
            pattern = re.compile(r'(?<=\\sphinxstyletheadfamily)(?P<value>[\s\S]*?)(?=(\\unskip|&)|\\\\)', re.M | re.I|re.U)
            aftercontent = headcontent
            #pattern = re.compile(r'(?<=\\sphinxstylethead{\\sphinxstyletheadfamily)([\s\S]*?)(?=\\unskip}\\relax &)', re.M | re.I|re.U)
        else:
            aftercontent = headcontent.replace(r'\\','&',1)
            pattern = re.compile(r'(?P<value>[\s\S]*?)(&\s{1})', re.M | re.I|re.U)
            #pattern = re.compile(r'[\s\S]*?&|\\unskip', re.M | re.I|re.U)

        mobjarr = pattern.finditer(aftercontent)
        headlist = []
        preposlist = []
        for mobj in mobjarr:
            amarr = mobj.group('value')
            curposlist = [mobj.start(),mobj.start()+len(amarr)]

            #用表头内容数组替换
            fontcolor = self.tablesattrobj.headfontcolor
            #先去掉首尾空格，避免首尾有空格无法去掉回车换行符
            amarr = amarr.strip()
            amarr = amarr.strip('\r')
            amarr = amarr.strip('\n')
            amarr = amarr.strip()  #再去掉首尾空格，避免多余的空格出现
            if amarr == '':
                continue
            fontcolor = fontcolor.replace('{}','{'+ amarr+'}',1)
            if len(preposlist) > 0:
                headlist.append(headcontent[preposlist[1]:curposlist[0]])
            else:
                headlist.append(headcontent[0:curposlist[0]])
            headlist.append(fontcolor)
            preposlist = curposlist

        headlist.append(headcontent[preposlist[1]:len(headcontent)])  #把最后一个字符串加上
        headcontent = ''
        for prelist in headlist:
            headcontent = headcontent + prelist + '\n'
        newcontent = content[0:posarr[0]]+r'\rowcolor'+headtype+'\n'+headcontent+content[posarr[1]:len(content)]
        return newcontent
    #渲染树型表格的第一列    
    def __ModifyVerticalTable(self, singletablecontent):
        #找出每一行的内容
        searchstr = r'(?<=\\hline)(?P<content>[\s\S]*?)(?=\\hline)'
        pattern = re.compile(searchstr,re.M | re.I|re.U)
        matchiter = pattern.finditer(singletablecontent)
        
        posarr=[]  #保存位置，便于组合
        i = 0
        for m in matchiter:
        
            posarr.append([])
            posarr[i] = m.span()
            
            if i ==0:
                newcontent = singletablecontent[0:posarr[i][0]]
            else:
                newcontent = newcontent+singletablecontent[posarr[i-1][1]:posarr[i][0]]
        
            cellcontent = m.group('content') #匹配到的第一个即为表头内容
            #将第一个单元格内容渲染成蓝底白字
            firstcellcontent = self.__ModifyFirstColumnType(cellcontent)
            newcontent += firstcellcontent 
            i+=1
        newcontent += singletablecontent[posarr[i-1][1]:len(singletablecontent)]
        return newcontent
            
    #渲染第一个单元格内容
    def __ModifyFirstColumnType(self,cellcontent):
    
        new_cellcontent = ""

        if r'\sphinxstyletheadfamily' in cellcontent:
        
            searchstr = r'(?<=\\sphinxstyletheadfamily)(?P<value>[\s\S]*?)(?=(\\unskip|&)|\\\\)'
            
            aftercontent = cellcontent.strip()
            aftercontent = aftercontent.strip('\r')
            aftercontent = aftercontent.strip('\n')
            aftercontent = aftercontent.strip()
            mobj = re.search(searchstr, aftercontent, re.M|re.I|re.U )  #匹配到的第一个既是需要修改的内容
            
            #修改字体颜色
            amarr = mobj.group('value')
            posarr = mobj.span()
            new_cellcontent = aftercontent[0:posarr[0]]+'\n'+r'\cellcolor'+self.tablesattrobj.headtype
            #用表头内容数组替换
            fontcolor = self.tablesattrobj.headfontcolor
            #先去掉首尾空格，避免首尾有空格无法去掉回车换行符
            amarr = amarr.strip()
            amarr = amarr.strip('\r')
            amarr = amarr.strip('\n')
            amarr = amarr.strip()  #再去掉首尾空格，避免多余的空格出现
            #if amarr == '':
            #    continue
            if (r'\textbf' or r'\textcolor') in amarr:
                return cellcontent
            fontcolor = fontcolor.replace('{}','{'+ amarr+'}',1)
            new_cellcontent +=r'{'+fontcolor + '}\n' + aftercontent[posarr[1]:len(aftercontent)]
        else:
            
            aftercontent = cellcontent.replace(r'\\','&',1)
            #去掉首尾空格和换行符
            aftercontent = aftercontent.strip()
            aftercontent = aftercontent.strip('\r')
            aftercontent = aftercontent.strip('\n')
            aftercontent = aftercontent.strip()
            
            tmplist = re.split(r'&',aftercontent)

            preposlist = 0
            #只对第一个做修改
            onelist = tmplist[0]
            
            #用表头内容数组替换
            fontcolor = self.tablesattrobj.headfontcolor
            #先去掉首尾空格，避免首尾有空格无法去掉回车换行符
            onelist = onelist.strip()
            onelist = onelist.strip('\r')
            onelist = onelist.strip('\n')
            onelist = onelist.strip()  #再去掉首尾空格，避免多余的空格出现
            #if onelist == '':
            #        continue
            if (r'\textbf' or r'\textcolor') in onelist:
                return cellcontent
            new_cellcontent = '\n'+r'\cellcolor'+self.tablesattrobj.headtype+r'{'+fontcolor.replace('{}','{'+ onelist+'}',1)+r'}'+'\n'

            for i in range(1,len(tmplist)):
                if len(tmplist[i])>0:
                    new_cellcontent += '&' +tmplist[i]
            new_cellcontent+=r'\\' #将最后被替换掉的\\再加上

        return new_cellcontent + '\n'

# 打开Makefile文件查找source和build文件夹
def OpenMakefile():
    global source_dir
    global build_dir
    source_dir = ''
    build_dir = ''
    try:
        if platform.system().lower() == 'windows':
            with open('make.bat',"r") as f:
                fstr = f.read()
                
            #用正则表达式查找source和build文件夹具体路径
            searchstr = r"set *SOURCEDIR *= *(\S+)"
            m = re.search(searchstr, fstr, re.M|re.I|re.U )
            source_dir = m.group(1) #匹配到的第一个即为source所在目录
            
            searchstr = r"set *BUILDDIR *= *(\S+)"
            m = re.search(searchstr, fstr, re.M|re.I|re.U )
            build_dir = m.group(1) #匹配到的第一个即为build所在目录
        else:
            with open('Makefile',"r") as f:
                fstr = f.read()
            
            #用正则表达式查找source和build文件夹具体路径
            searchstr = r"SOURCEDIR *= *(\S+)"
            m = re.search(searchstr, fstr, re.M|re.I|re.U )
            source_dir = m.group(1) #匹配到的第一个即为源所在目录
            
            searchstr = r"BUILDDIR *= *(\S+)"
            m = re.search(searchstr, fstr, re.M|re.I|re.U )
            build_dir = m.group(1) #匹配到的第一个即为源所在目录
        
    except Exception as e:
        print(e)
        return

def GetLatex_documents():
    global source_dir
    if source_dir == '':
        return
    #得到配置文件conf.py的路径
    if source_dir == '.':
        confdir = './conf.py'
    else:
        confdir = './' + source_dir +'/conf.py'

    conffile = os.path.abspath(confdir)
     #打开conf.py文件
    with codecs.open(conffile,"r+",encoding='utf-8') as f:
            fstr = f.read()
    list = []
    versioninfo = GetVersionInfo(fstr)
    fileprefile = GetFilePreInfo(fstr)
    if versioninfo=="" or fileprefile=="":
    #根据正则表达式，找出latex_documents内容
        searchstr = r"latex_documents *= *\[([\s\S]*?)\]"
        m = re.search(searchstr, fstr, re.M|re.I|re.U )
        latex_documents = m.group(1) #匹配到的第一个即为源所在目录
        #拆分二维数组，兼容多个情况
        list = latex_documents.split(")")
        for i in range(len(list)):
            if IsComment(list[i]):
                list[i]= list[i].split(",")
        list.pop()
    else:
        #创建2维列表
        for i in range(1):
            list.append([])
            for j in range(2):
                list[i].append("")
        list[0][0] = "comment"  #为了兼容文件解析内容，进行补位。
        list[0][1] = '"' + fileprefile + versioninfo + '.tex"'  #为了兼容解析过程，添加双引号。
    return list

#得到版本信息
def GetVersionInfo(fstr):
    
    versioninfo = ""
    searchstr = r"version *= *(u*)'(.*?)'"
    m = re.search(searchstr, fstr, re.I)
    if m != None:
        versioninfo = m.group(2) #匹配到的第一个即为源所在目录
    print("version = " + versioninfo)
    return versioninfo

#得到文件前缀信息
def GetFilePreInfo(fstr):
    filepre = ""
    searchstr = r"curfnpre *= *(u*)'(.*?)'"
    m = re.search(searchstr, fstr, re.M|re.I|re.U )
    if m != None:
        filepre = m.group(2) #匹配到的第一个即为源所在目录
    print("filepre = " + filepre)
    return filepre

#判断是否为注释行
def IsComment(instr):

    if instr.strip() is None:
        return False

    rule = re.compile('^#.*$')
    if rule.match(instr.strip()) is None:
        return True;
    else:
        return False;

#根据正则表达式取单引号和双引号中的内容
def getquomarkcontent(strarr):
    #根据正则表达式，找出双引号和单引号中的内容
    print("texfile="+strarr)
    searchstr = r"[\"|'](.*?)[\"|']"
    m = re.search(searchstr, strarr, re.M|re.I|re.U )
    if m is None:
        return None
    return m.group(1).strip() #匹配到的第一个即为源所在目录

source_dir = '' #保存源文件所在目录
build_dir = ''  #保存生成文件所在目录
OpenMakefile()
latex_documents = GetLatex_documents() #保存latex文档所在数组
__load_dict__ = __openconfjsonfile__()
if __name__ == '__main__':

    latexdir = ''
    # 可以自定义latex目录，通过参数传入，如果传入参数默认是latex目录
    if len(sys.argv) > 1:
        latexdir = '/' + sys.argv[1] + '/'
    else:
        latexdir = '/latex/'
        
    doclen = len(latex_documents)
    for i in range(0,doclen):
        #得到latex路径
        latexpath = './' + build_dir + latexdir
        #copy 背景图到latex路径，背景图必须与该文件在同一个目录下
        if os.path.exists('./chapterbkpaper.pdf'):
            shutil.copy('./chapterbkpaper.pdf',latexpath)
        #得到相对路径
        if getquomarkcontent(latex_documents[i][1]) is None:
            continue
        texfilepath = latexpath + getquomarkcontent(latex_documents[i][1])
        #相对路径转绝对路径
        texfile = os.path.abspath(texfilepath)
        if not os.path.exists(texfile):
            continue
        fo = codecs.open(texfile, "r+",encoding = 'utf-8')
        texcontent = fo.read()
        fo.close
        
        #得到修改tex文件的对象
        ModTexobj = clsModifyTex(texcontent)
        ModTexobj.AddPackageToTex()
        ModTexobj.AddOtherTocToTex()
        ModTexobj.AddCustormOptionsToTex()
        ModTexobj.ModifyReplacePackage()
        ModTexobj.ModifyTablesAttributes()
        
        fw = codecs.open(texfile, "w+",encoding = 'utf-8')
        fw.write(ModTexobj.content)
        fw.close



