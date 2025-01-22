import os
import typer
from markdown import markdown
import re

app = typer.Typer()

REPORT_FILE = "report.md"
HTML_OUTPUT_FILE = "report.html"


@app.command()
def html():
    """
    Generate an HTML page from the markdown report.
    """
    if not os.path.exists(REPORT_FILE):
        typer.echo(f"Error: {REPORT_FILE} not found in the current directory.")
        raise typer.Exit()

    try:
        with open(REPORT_FILE, "r", encoding="utf-8") as f:
            markdown_content = f.read()

        html_content = markdown(markdown_content)

        with open(HTML_OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(html_content)

        typer.echo(f"HTML report generated successfully: {HTML_OUTPUT_FILE}")

    except Exception as e:
        typer.echo(f"An error occurred while generating the HTML report: {e}")
        raise typer.Exit()


@app.command()
def check():
    """
    Check the markdown report for constraints like missing answers or images.
    """
    if not os.path.exists(REPORT_FILE):
        typer.echo(f"Error: {REPORT_FILE} not found in the current directory.")
        raise typer.Exit()

    try:
        with open(REPORT_FILE, "r", encoding="utf-8") as f:
            content = f.read()

        issues = []

        # Count total questions and unanswered questions
        total_questions = content.count("--- question")
        unanswered_questions = len(re.findall(r"--- question.*?fill here.*?---", content, flags=re.DOTALL))
        empty_answers = total_questions - unanswered_questions

        if empty_answers > 0:
            issues.append(f"There are {empty_answers} unanswered questions in the report.")

        # Check for missing images where required
        if "![my_image](figures/" not in content:
            issues.append("Some image answers may be missing.")

        if issues:
            typer.echo("Issues found in the report:")
            for issue in issues:
                typer.echo(f"- {issue}")
        else:
            typer.echo("The report passed all checks!")

    except Exception as e:
        typer.echo(f"An error occurred while checking the report: {e}")
        raise typer.Exit()



if __name__ == "__main__":
    app()
