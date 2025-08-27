

from flask import Flask, render_template
import markdown

app = Flask(__name__)

@app.route('/')
def index():
    with open('README.md', encoding='utf-8') as f:
        readme_content = f.read()
    readme_html = markdown.markdown(readme_content, extensions=['fenced_code', 'tables'])
    return render_template('index.html', readme_html=readme_html)

if __name__ == '__main__':
    app.run(debug=True)
