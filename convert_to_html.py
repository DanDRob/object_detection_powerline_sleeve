import markdown
import os

def convert_md_to_html(md_file, html_file):
    """Convert markdown to a styled HTML file"""
    # Read markdown content
    with open(md_file, 'r') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'codehilite', 'toc']
    )
    
    # Add some basic CSS for better formatting
    html_with_css = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Powerline Sleeve Detection Project Guide</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 40px; 
                line-height: 1.6;
                color: #333;
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
            }}
            code {{ 
                background-color: #f5f5f5; 
                padding: 2px 5px; 
                border-radius: 3px;
                font-family: Consolas, Monaco, 'Andale Mono', monospace;
                font-size: 90%;
            }}
            pre {{ 
                background-color: #f5f5f5; 
                padding: 15px; 
                border-radius: 5px; 
                overflow-x: auto;
                border: 1px solid #ddd;
            }}
            h1 {{ 
                color: #2c3e50; 
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{ 
                color: #34495e; 
                border-bottom: 1px solid #eee; 
                padding-bottom: 5px;
                margin-top: 30px;
            }}
            h3 {{ 
                color: #34495e; 
                margin-top: 25px;
            }}
            table {{ 
                border-collapse: collapse; 
                width: 100%; 
                margin: 20px 0;
                border: 1px solid #ddd;
            }}
            th, td {{ 
                border: 1px solid #ddd; 
                padding: 8px; 
                text-align: left;
            }}
            th {{ 
                background-color: #f5f5f5; 
            }}
            a {{
                color: #3498db;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            .page-break {{
                page-break-after: always;
            }}
            ul, ol {{
                padding-left: 25px;
            }}
            @media print {{
                body {{
                    margin: 0;
                    padding: 0;
                }}
                pre, code {{
                    white-space: pre-wrap;
                }}
                a[href]:after {{
                    content: " (" attr(href) ")";
                    font-size: 90%;
                    color: #666;
                }}
                .no-print, .no-print * {{
                    display: none !important;
                }}
                h1, h2, h3, h4, h5, h6 {{
                    page-break-after: avoid;
                }}
                table, figure {{
                    page-break-inside: avoid;
                }}
            }}
        </style>
    </head>
    <body>
        {html_content}
        <script>
            // Add print button
            window.onload = function() {{
                // Create print instructions div
                var printInstructions = document.createElement('div');
                printInstructions.className = 'no-print';
                printInstructions.style.backgroundColor = '#f0f8ff';
                printInstructions.style.padding = '10px';
                printInstructions.style.margin = '20px 0';
                printInstructions.style.borderRadius = '5px';
                printInstructions.style.border = '1px solid #add8e6';
                
                printInstructions.innerHTML = `
                    <h3>To create a PDF:</h3>
                    <ol>
                        <li>Click the "Print to PDF" button below</li>
                        <li>In the print dialog, select "Save as PDF" as the destination</li>
                        <li>Click "Save" and choose where to save the PDF file</li>
                    </ol>
                    <button onclick="window.print()" style="padding: 10px 20px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer;">Print to PDF</button>
                `;
                
                // Insert at the beginning of the body
                document.body.insertBefore(printInstructions, document.body.firstChild);
            }};
        </script>
    </body>
    </html>
    """
    
    # Create HTML file
    with open(html_file, 'w') as f:
        f.write(html_with_css)
    
    print(f"HTML file created: {html_file}")
    print("To generate a PDF:")
    print(f"1. Open the HTML file: {html_file}")
    print("2. Use your browser's 'Print' function (Ctrl+P or Cmd+P)")
    print("3. Select 'Save as PDF' as the destination")
    print("4. Save the file as 'project_guide.pdf'")

if __name__ == "__main__":
    convert_md_to_html("project_guide.md", "project_guide.html") 