#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para converter a explicação da Base 100 de Markdown para PDF
"""

import markdown
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import re
import os

def markdown_to_pdf(markdown_file, pdf_file):
    """
    Converte arquivo Markdown para PDF com formatação personalizada
    """
    
    # Ler o arquivo Markdown
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Criar documento PDF
    doc = SimpleDocTemplate(pdf_file, pagesize=A4,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)
    
    # Estilos
    styles = getSampleStyleSheet()
    
    # Estilos personalizados
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.darkblue
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=10,
        spaceBefore=15,
        textColor=colors.darkgreen
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        alignment=TA_JUSTIFY
    )
    
    code_style = ParagraphStyle(
        'CustomCode',
        parent=styles['Code'],
        fontSize=10,
        spaceAfter=10,
        spaceBefore=10,
        backColor=colors.lightgrey,
        borderColor=colors.grey,
        borderWidth=1,
        borderPadding=5
    )
    
    # Lista para armazenar elementos do PDF
    story = []
    
    # Processar o conteúdo Markdown linha por linha
    lines = markdown_content.split('\n')
    
    for line in lines:
        line = line.strip()
        
        if not line:
            story.append(Spacer(1, 6))
            continue
            
        # Título principal
        if line.startswith('# '):
            title = line[2:].strip()
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 20))
            
        # Cabeçalhos H2
        elif line.startswith('## '):
            heading = line[3:].strip()
            story.append(Paragraph(heading, heading_style))
            
        # Cabeçalhos H3
        elif line.startswith('### '):
            subheading = line[4:].strip()
            story.append(Paragraph(subheading, subheading_style))
            
        # Cabeçalhos H4
        elif line.startswith('#### '):
            subheading = line[5:].strip()
            story.append(Paragraph(subheading, subheading_style))
            
        # Código
        elif line.startswith('```'):
            continue  # Ignorar marcadores de código
            
        # Listas
        elif line.startswith('- ') or line.startswith('* '):
            item = line[2:].strip()
            story.append(Paragraph(f"• {item}", normal_style))
            
        # Tabelas (formato simples)
        elif '|' in line and not line.startswith('|---'):
            # Processar tabela simples
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if cells:
                story.append(Paragraph(' | '.join(cells), normal_style))
                
        # Texto normal
        elif line and not line.startswith('|---'):
            # Remover emojis e formatação markdown básica
            clean_line = re.sub(r'[📊🎯📈🔢🌟✅🔴🟢⚪🔍📚📈🎯]', '', line)
            clean_line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_line)
            clean_line = re.sub(r'\*(.*?)\*', r'<i>\1</i>', clean_line)
            clean_line = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', clean_line)
            
            if clean_line.strip():
                story.append(Paragraph(clean_line, normal_style))
    
    # Adicionar tabela de exemplo
    story.append(Spacer(1, 20))
    story.append(Paragraph("Exemplo de Comparação:", heading_style))
    
    # Dados da tabela
    table_data = [
        ['Ativo', 'Preço Inicial', 'Preço Final', 'Base 100', 'Performance'],
        ['SPY', '$400', '$440', '110', '+10%'],
        ['GLD', '$180', '$190', '105.6', '+5.6%'],
        ['VIX', '$20', '$18', '90', '-10%']
    ]
    
    # Criar tabela
    table = Table(table_data, colWidths=[1*inch, 1.2*inch, 1.2*inch, 1*inch, 1*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    story.append(Spacer(1, 20))
    
    # Rodapé
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        alignment=TA_CENTER,
        textColor=colors.grey
    )
    
    story.append(Spacer(1, 30))
    story.append(Paragraph("Documento gerado automaticamente", footer_style))
    story.append(Paragraph("Projeto Big Data - Análise de Eventos Financeiros", footer_style))
    
    # Gerar PDF
    doc.build(story)
    print(f"✅ PDF criado com sucesso: {pdf_file}")

if __name__ == "__main__":
    # Arquivos de entrada e saída
    markdown_file = "explicacao_base_100.md"
    pdf_file = "Explicacao_Base_100_Normalizacao.pdf"
    
    # Verificar se o arquivo Markdown existe
    if not os.path.exists(markdown_file):
        print(f"❌ Arquivo não encontrado: {markdown_file}")
        exit(1)
    
    try:
        # Converter para PDF
        markdown_to_pdf(markdown_file, pdf_file)
        print(f"📄 PDF salvo em: {os.path.abspath(pdf_file)}")
        
    except Exception as e:
        print(f"❌ Erro ao gerar PDF: {e}")