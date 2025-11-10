from faker import Faker
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
import datetime

def generate_dummy_pdf(file_path):
    fake = Faker()
    c = canvas.Canvas(file_path, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 100, "Company Compliance Conversation Log")
    
    # Metadata
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 150, f"User Name: {fake.name()}")
    c.drawString(100, height - 170, f"User Email: {fake.email()}")
    c.drawString(100, height - 190, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(100, height - 210, f"Conversation ID: {fake.uuid4()}")
    
    # Compliance Status
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, height - 250, "Compliance Status: Compliant")
    
    # Conversation Log
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, height - 300, "Conversation Log:")
    
    text = c.beginText(100, height - 320)
    text.setFont("Helvetica", 10)
    
    for _ in range(5):
        text.textLine(fake.sentence(nb_words=25))

    c.drawText(text)
    c.save()

if __name__ == "__main__":
    output_dir = "./pdfs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i in range(3):
        file_path = os.path.join(output_dir, f"dummy_log_{i+1}.pdf")
        generate_dummy_pdf(file_path)
        print(f"Generated {file_path}")
