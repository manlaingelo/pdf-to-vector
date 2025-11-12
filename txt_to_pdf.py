#!/usr/bin/env python3
"""
Text to PDF Converter
Converts dummy1.txt to a PDF file using reportlab
"""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import os


def txt_to_pdf(input_file, output_file):
    """
    Convert a text file to PDF
    
    Args:
        input_file: Path to input text file
        output_file: Path to output PDF file
    """
    # Create a canvas
    c = canvas.Canvas(output_file, pagesize=letter)
    width, height = letter
    
    # Set up text parameters
    margin = 0.75 * inch
    y_position = height - margin
    line_height = 14
    max_width = width - (2 * margin)
    
    # Read the text file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Add text to PDF
    c.setFont("Helvetica", 10)
    
    for line in lines:
        line = line.rstrip('\n')
        
        # Check if we need a new page
        if y_position < margin:
            c.showPage()
            c.setFont("Helvetica", 10)
            y_position = height - margin
        
        # Handle long lines by wrapping
        if len(line) > 0:
            # Simple word wrapping
            words = line.split()
            current_line = ""
            
            for word in words:
                test_line = current_line + " " + word if current_line else word
                # Approximate width check (more accurate would use stringWidth)
                if c.stringWidth(test_line, "Helvetica", 10) < max_width:
                    current_line = test_line
                else:
                    if current_line:
                        c.drawString(margin, y_position, current_line)
                        y_position -= line_height
                        if y_position < margin:
                            c.showPage()
                            c.setFont("Helvetica", 10)
                            y_position = height - margin
                    current_line = word
            
            if current_line:
                c.drawString(margin, y_position, current_line)
                y_position -= line_height
        else:
            # Empty line
            y_position -= line_height
    
    # Save the PDF
    c.save()
    print(f"PDF created successfully: {output_file}")


if __name__ == "__main__":
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define input and output paths
    input_file = os.path.join(script_dir, "raw_texts", "dummy1.txt")
    output_file = os.path.join(script_dir, "dummy1.pdf")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        exit(1)
    
    # Convert to PDF
    txt_to_pdf(input_file, output_file)
