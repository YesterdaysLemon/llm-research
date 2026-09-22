"""Render the authored Markdown paper using ReportLab; inspect page PNGs afterward."""
import argparse,html,re
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
fontroot=Path('C:/Windows/Fonts')
for name,file in [('Body','calibri.ttf'),('Body-Bold','calibrib.ttf'),('Body-Italic','calibrii.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(fontroot/file)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Body-Bold',italic='Body-Italic',boldItalic='Body-Bold')
styles=getSampleStyleSheet()
for name in ['Normal','BodyText']:
    styles[name].fontName='Body';styles[name].fontSize=10.5;styles[name].leading=14.3;styles[name].spaceAfter=7
styles['Title'].fontName='Body-Bold';styles['Title'].fontSize=27;styles['Title'].leading=30;styles['Title'].alignment=TA_LEFT;styles['Title'].spaceAfter=12
for name,size in [('Heading1',15),('Heading2',12)]:
    styles[name].fontName='Body-Bold';styles[name].fontSize=size;styles[name].leading=size+3;styles[name].textColor=colors.HexColor('#164b53');styles[name].spaceBefore=11;styles[name].spaceAfter=6
    styles[name].keepWithNext=True
styles.add(ParagraphStyle('Quote',parent=styles['BodyText'],fontName='Body-Italic',leftIndent=12,borderColor=colors.HexColor('#abc4bd'),borderWidth=0.5,borderPadding=7,spaceBefore=5,spaceAfter=10))
styles.add(ParagraphStyle('SmallCell',parent=styles['BodyText'],fontSize=8.8,leading=11,spaceAfter=0))
def inline(s):
    s=html.escape(s.replace('\u2011','-').replace('\u2013','-').replace('\u2014','-'))
    s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:f'<a href="{m.group(2)}" color="#166677">{m.group(1)}</a>',s)
    s=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'`([^`]+)`',r'<font name="Courier" size="9">\1</font>',s)
    return s
lines=a.source.read_text(encoding='utf8').splitlines();story=[];i=0;width=6.8*inch
while i<len(lines):
    line=lines[i].strip();i+=1
    if not line:continue
    if line=='<!-- pagebreak -->':story.append(PageBreak());continue
    if line.startswith('!['):
        m=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line);p=a.source.parent/m[2]
        img=Image(str(p));img.drawHeight*=width/img.drawWidth;img.drawWidth=width;story += [img,Spacer(1,7)];continue
    if line.startswith('|'):
        tablelines=[line]
        while i<len(lines) and lines[i].strip().startswith('|'):tablelines.append(lines[i].strip());i+=1
        values=[[Paragraph(inline(x.strip()),styles['SmallCell']) for x in s.strip('|').split('|')] for s in tablelines if not re.fullmatch(r'[| :\-]+',s)]
        n=len(values[0]);weights=([1,2.2] if 'Condition' in tablelines[0] and n==2 else [1.6]+[1]*(n-1));cols=[width*w/sum(weights) for w in weights]
        t=Table(values,colWidths=cols,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5efed')),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#819e98')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#dce4e2')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story += [t,Spacer(1,9)];continue
    style='BodyText'
    if line.startswith('# '):style='Title';line=line[2:]
    elif line.startswith('## '):style='Heading1';line=line[3:]
    elif line.startswith('### '):style='Heading2';line=line[4:]
    elif line.startswith('> '):style='Quote';line=line[2:]
    elif line.startswith('- '):line='&#8226; '+inline(line[2:]);story.append(Paragraph(line,styles[style]));continue
    else:
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','>','![','<!--','- ')):
            line+=' '+lines[i].strip();i+=1
    story.append(Paragraph(inline(line),styles[style]))
def footer(c,doc):
    c.setStrokeColor(colors.HexColor('#bccdc8'));c.line(.85*inch,.59*inch,7.65*inch,.59*inch)
    c.setFont('Body',8);c.setFillColor(colors.HexColor('#526c68'))
    c.drawString(.85*inch,.40*inch,'E009  |  Exploratory research note  |  September 2026')
    c.drawRightString(7.65*inch,.40*inch,str(doc.page))
a.output.parent.mkdir(parents=True,exist_ok=True)
doc=SimpleDocTemplate(str(a.output),pagesize=(8.5*inch,11*inch),rightMargin=.85*inch,leftMargin=.85*inch,topMargin=.65*inch,bottomMargin=.8*inch,title='A voice without a prescribed self',author='Codex, prepared for Alireza')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(a.output)
