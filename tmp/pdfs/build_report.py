from pathlib import Path
import re
from html import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/pdf/Modular_RAG_Azure_Project_Report.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
for name, file in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf'),('Arial-Italic','ariali.ttf'),('Arial-BoldItalic','arialbi.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(Path('C:/Windows/Fonts') / file)))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial-Italic',boldItalic='Arial-BoldItalic')
navy=colors.HexColor('#18334B'); teal=colors.HexColor('#167D8D')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Body',fontName='Arial',fontSize=10,leading=15,spaceAfter=8,textColor=navy))
styles.add(ParagraphStyle(name='Section',parent=styles['Body'],fontName='Arial-Bold',fontSize=16,leading=21,spaceBefore=18,spaceAfter=10,keepWithNext=True))
styles.add(ParagraphStyle(name='Subsection',parent=styles['Body'],fontName='Arial-Bold',fontSize=11,leading=16,spaceBefore=10,spaceAfter=6,keepWithNext=True))
styles.add(ParagraphStyle(name='Cell',parent=styles['Body'],fontSize=8.6,leading=12,spaceAfter=0))
styles.add(ParagraphStyle(name='CellHead',parent=styles['Cell'],fontName='Arial-Bold',textColor=colors.white))
styles.add(ParagraphStyle(name='BulletBody',parent=styles['Body'],leftIndent=12,firstLineIndent=-9,spaceAfter=5))
styles.add(ParagraphStyle(name='CoverTitle',parent=styles['Body'],fontName='Arial-Bold',fontSize=34,leading=42,spaceAfter=18))
styles.add(ParagraphStyle(name='CoverSub',parent=styles['Body'],fontSize=19,leading=27,textColor=teal,spaceAfter=22))

def inline(text):
    text=text.replace('—',' - ').replace('–','-').replace('\u2011','-')
    text=escape(text)
    text=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'`([^`]+)`',r'\1',text)
    return text

class Report(SimpleDocTemplate):
    def afterFlowable(self,f):
        if isinstance(f,Paragraph) and f.style.name=='Section':
            t=f.getPlainText()
            if re.match(r'^\d+\.',t):
                key='section-'+t.split('.')[0]
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(t,key,0,False)
                self.notify('TOCEntry',(0,t,self.page,key))

def footer(c,doc):
    w,h=A4
    c.saveState()
    if doc.page>1:
        c.setStrokeColor(colors.HexColor('#D8E4E9'));c.line(44,h-35,w-44,h-35)
        c.setFont('Arial',8);c.setFillColor(teal)
        c.drawString(44,h-25,'MODULAR-RAG AZURE  /  PROJECT REPORT')
    c.setFillColor(navy);c.setFont('Arial',8)
    c.drawString(44,26,'Plain-English guide  |  18 October 2026')
    c.drawRightString(w-44,26,str(doc.page))
    c.restoreState()

story=[Spacer(1,85),Paragraph('Modular-RAG<br/>Azure',styles['CoverTitle']),Paragraph('A plain-English project report',styles['CoverSub']),Paragraph('What we use, why we use it, and how the complete application works.',styles['Body']),Spacer(1,32),Paragraph('<b>Prepared:</b> 18 October 2026',styles['Body']),Paragraph('<b>For:</b> Readers with no programming or cloud background',styles['Body']),Paragraph('<b>Based on:</b> The application code, interface, tests, and evaluation files in this project.',styles['Body']),Spacer(1,26),Paragraph('A guide to document upload, four search methods, AI answers, source checking, storage, testing, and current limitations.',styles['Body']),PageBreak(),Paragraph('Contents',styles['Section'])]
toc=TableOfContents()
toc.levelStyles=[ParagraphStyle(name='TOC',fontName='Arial',fontSize=10,leading=12,spaceBefore=0,textColor=navy)]
story.extend([toc,PageBreak()])
lines=(ROOT/'PROJECT_REPORT_SIMPLE_ENGLISH.md').read_text(encoding='utf-8').splitlines()
start=next(i for i,l in enumerate(lines) if l.startswith('This report explains'))
i=start
while i<len(lines):
    line=lines[i].strip()
    if not line:
        i+=1;continue
    if line.startswith('## '):
        story.append(Paragraph(inline(line[3:]),styles['Section']));i+=1;continue
    if line.startswith('### '):
        story.append(Paragraph(inline(line[4:]),styles['Subsection']));i+=1;continue
    if line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            cells=[s.strip() for s in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[-: ]+',s) for s in cells): rows.append(cells)
            i+=1
        n=len(rows[0]);width=A4[0]-88
        widths=[width*.30,width*.70] if n==2 else [width*.23,width*.37,width*.40]
        cells=[[Paragraph(inline(s),styles['CellHead' if r==0 else 'Cell']) for s in row] for r,row in enumerate(rows)]
        table=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#EFF5F7'),colors.white]),('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#D8E4E9'))]))
        story.extend([table,Spacer(1,12)]);continue
    if line.startswith('- '):
        story.append(Paragraph('• '+inline(line[2:]),styles['BulletBody']));i+=1;continue
    if re.match(r'^\d+\. ',line):
        story.append(Paragraph(inline(line),styles['BulletBody']));i+=1;continue
    para=[line];i+=1
    while i<len(lines) and lines[i].strip() and not re.match(r'^(#|\||- |\d+\. )',lines[i].strip()):
        para.append(lines[i].strip());i+=1
    story.append(Paragraph(inline(' '.join(para)),styles['Body']))

doc=Report(str(OUT),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=51,bottomMargin=48,title='Modular-RAG Azure - Plain-English Project Report',author='Modular-RAG Azure',allowSplitting=1)
doc.multiBuild(story,onFirstPage=footer,onLaterPages=footer)
reader=PdfReader(OUT)
alltext='\n'.join(p.extract_text() or '' for p in reader.pages)
for title in re.findall(r'^## (\d+\..+)$','\n'.join(lines),re.M):
    assert title in alltext, title
assert '1,536' in alltext and '25. A short explanation' in alltext
print(f'Created {OUT}; pages={len(reader.pages)}; bytes={OUT.stat().st_size}')

import pypdfium2 as pdfium
from PIL import Image,ImageOps,ImageDraw
pdf=pdfium.PdfDocument(str(OUT))
thumbs=[]
for p in range(len(pdf)):
    img=pdf[p].render(scale=1.2).to_pil().convert('RGB')
    img.save(ROOT/f'tmp/pdfs/page-{p+1:02d}.png')
    img.thumbnail((298,421))
    tile=Image.new('RGB',(318,451),'#D9E1E7');tile.paste(img,((318-img.width)//2,8))
    ImageDraw.Draw(tile).text((12,432),str(p+1),fill='black')
    thumbs.append(tile)
for batch in range(0,len(thumbs),8):
    chunk=thumbs[batch:batch+8]
    sheet=Image.new('RGB',(318*4,451*((len(chunk)+3)//4)),'white')
    for j,tile in enumerate(chunk):sheet.paste(tile,((j%4)*318,(j//4)*451))
    sheet.save(ROOT/f'tmp/pdfs/contact-{batch//8+1}.png')


