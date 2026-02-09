import os
import sys
import logging
import traceback
import contextlib
from datetime import datetime
from typing import List, Dict, Any, Union

# --- LangChain & QGenie Imports ---
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from qgenie.integrations.langchain import QGenieChat
from qgenie_sdk_tools.tools.email import email_tool

# --- Environment Configuration ---
try:
    from utils.parsers.env_parser import EnvConfig
except ImportError:
    class EnvConfig:
        def __init__(self):
            self.config = {
                'LLM_MODEL': os.getenv('LLM_MODEL', 'Turbo'),
                'QGENIE_API_KEY': os.getenv('QGENIE_API_KEY', '')
            }

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Context Manager to Silence External Prints ---
@contextlib.contextmanager
def suppress_output():
    """Redirects stdout and stderr to devnull to silence external library prints."""
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

class CodebaseEmailReporter:
    """
    Unified Email Reporter for Codebase Analysis & Fixer Agents.
    """
    
    MAX_ATTACHMENT_SIZE = 15 * 1024 * 1024  # 15 MB Limit

    def __init__(self):
        self.logger = logging.getLogger("CodebaseEmailReporter")
        
        # Initialize Config
        self.env = EnvConfig()
        self.model_name = self.env.config.get('LLM_MODEL', 'Turbo')
        self.api_key = self.env.config.get('QGENIE_API_KEY')
        
        try:
            # Initialize LLM
            self.llm = QGenieChat(
                model=self.model_name,
                max_tokens=4096, 
                temperature=0.0, 
                api_key=self.api_key,
                timeout=60000 
            )
        except Exception as e:
            self.logger.error(f"Failed to init QGenieChat: {e}")

    def send_report(
        self, 
        recipients: Union[str, List[str]], 
        attachment_path: str, 
        metadata: Dict[str, Any], 
        stats: Dict[str, Any], 
        analysis_summary: str
    ) -> bool:
        """
        Generates the HTML report and invokes the QGenie Agent to send it.
        """

        # 1. Validate Recipients
        if not recipients:
            self.logger.error("No recipients provided.")
            return False
            
        if isinstance(recipients, list):
            recipients_str = ",".join(str(r).strip() for r in recipients)
        else:
            recipients_str = str(recipients).strip()

        # 2. Validate Attachment
        final_attachment_path = None
        if attachment_path:
            if os.path.exists(attachment_path):
                size = os.path.getsize(attachment_path)
                if size > self.MAX_ATTACHMENT_SIZE:
                    self.logger.warning(f"Attachment too large ({size} bytes). Skipping.")
                    final_attachment_path = None
                else:
                    final_attachment_path = attachment_path
            else:
                self.logger.warning(f"Attachment not found: {attachment_path}")
                final_attachment_path = None

        # 3. Generate HTML Content
        try:
            html_body = self._generate_html(metadata, stats, analysis_summary)
        except Exception as e:
            self.logger.error(f"Error generating HTML: {e}")
            return False

        # 4. Invoke Agent (Wrapped in silencer)
        # We suppress output here because the SDK prints status messages to stdout
        with suppress_output():
            return self._invoke_email_agent(recipients_str, html_body, final_attachment_path)

    def _invoke_email_agent(self, recipients: str, html_body: str, attachment: str) -> bool:
        """
        Sets up the LangChain Agent with email_tool and executes the request.
        """
        
        # Prepare exact arguments for the tool
        tool_input = {
            "email_address": recipients,
            "subject": f"Codebase Analysis Report - {datetime.now().strftime('%Y-%m-%d')}",
            "body": html_body,
            "attachment": attachment, 
            "is_html": True
        }
        
        try:
            tools = [email_tool]

            # 1. Primary Method: Agent Executor
            system_prompt = (
                "You are an AI Email Assistant. "
                "You have access to a tool called 'email_tool'. "
                "When asked to send an email, you MUST generate a tool call for 'email_tool' using the EXACT arguments provided in the prompt. "
                "Do NOT generate text. ONLY generate the tool call."
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", 
                 "Call the email_tool function with these EXACT parameters:\n"
                 "email_address: {recipients}\n"
                 "subject: {subject}\n"
                 "attachment: {attachment}\n"
                 "body: <Provided in variable below>\n\n"
                 "HTML Body Content:\n{html_body}\n\n"
                 "EXECUTE TOOL NOW."
                ),
                ("placeholder", "{agent_scratchpad}"),
            ])

            agent = create_tool_calling_agent(self.llm, tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent, 
                tools=tools, 
                verbose=False,
                handle_parsing_errors=True,
                return_intermediate_steps=True 
            )

            result = agent_executor.invoke({
                "recipients": tool_input["email_address"],
                "subject": tool_input["subject"],
                "attachment": str(tool_input["attachment"]) if tool_input["attachment"] else "None",
                "html_body": tool_input["body"]
            })
            
            # 2. Verify if the tool was actually executed
            steps = result.get("intermediate_steps", [])
            tool_called = False
            
            for step in steps:
                if isinstance(step, tuple) and len(step) > 0:
                    action = step[0]
                    if getattr(action, 'tool', '') == 'email_tool':
                        tool_called = True

            if tool_called:
                return True
                
            # 3. Fallback: Direct Invocation
            direct_result = email_tool.invoke(tool_input)
            return True

        except Exception as e:
            # We print error to stderr explicitly so it might still show up if only stdout is suppressed,
            # but since we wrapped both in suppress_output, this will also be hidden unless we change the wrapper.
            # Assuming silence is desired for success path only.
            self.logger.error(f"Failed to send email: {e}")
            return False

    def _generate_html(self, metadata: Dict, stats: Dict, analysis: str) -> str:
        """
        Generates a clean, professional HTML report.
        """
        today = datetime.today().strftime('%Y-%m-%d %H:%M')
        
        # --- Stats Grid Generation ---
        stats_items = ""
        if stats:
            for k, v in stats.items():
                k_lower = k.lower()
                if any(x in k_lower for x in ['fail', 'error', 'bug', 'critical']):
                    border_color = "#e74c3c" # Red
                    icon = "🐞"
                elif any(x in k_lower for x in ['pass', 'fix', 'success', 'scanned']):
                    border_color = "#27ae60" # Green
                    icon = "✅"
                else:
                    border_color = "#3498db" # Blue
                    icon = "ℹ️"
                
                stats_items += f"""
                <div style="background:#fff; padding:12px; border-radius:6px; border-left: 4px solid {border_color}; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align:center;">
                    <div style="font-size:11px; text-transform:uppercase; color:#7f8c8d; font-weight:bold; letter-spacing:0.5px;">{k}</div>
                    <div style="font-size:20px; font-weight:700; color:#2c3e50; margin-top:4px;">{icon} {v}</div>
                </div>
                """

        # --- Metadata Table Generation ---
        meta_rows = ""
        for k, v in metadata.items():
            meta_rows += f"<tr><td style='padding:6px 10px; border-bottom:1px solid #eee; color:#666; width:40%;'>{k}</td><td style='padding:6px 10px; border-bottom:1px solid #eee; font-weight:600; color:#333;'>{v}</td></tr>"

        # --- Analysis Text Formatting ---
        if analysis:
            analysis_html = analysis.replace("\n", "<br/>")
            analysis_section = f"""
            <div style="background:#f8f9fa; padding:15px; border-radius:6px; border:1px solid #e9ecef; margin-top:20px;">
                <h3 style="margin-top:0; margin-bottom:10px; font-size:16px; color:#2c3e50; border-bottom:1px solid #ddd; padding-bottom:8px;">🔍 Executive Summary</h3>
                <div style="font-size:14px; line-height:1.6; color:#444;">{analysis_html}</div>
            </div>
            """
        else:
            analysis_section = ""

        # --- Full HTML Template ---
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
        </head>
        <body style="font-family: 'Segoe UI', Helvetica, Arial, sans-serif; background-color: #f4f6f8; margin: 0; padding: 20px;">
            <div style="max-width: 700px; margin: 0 auto; background: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
                
                <!-- Header -->
                <div style="background: linear-gradient(135deg, #2c3e50 0%, #4a698a 100%); padding: 20px; text-align: center; color: white;">
                    <h1 style="margin:0; font-size:22px;">Codebase Analysis Report</h1>
                    <div style="margin-top:5px; font-size:13px; opacity:0.9;">Generated by QGenie Agent</div>
                </div>
                
                <div style="padding: 25px;">
                    <p style="color:#555; font-size:14px; line-height:1.5;">
                        Hello,<br><br>
                        The automated codebase analysis is complete. Please find the summary of results below.
                        A detailed line-by-line report is available in the attached Excel file.
                    </p>
                    
                    <!-- Statistics Grid -->
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 20px;">
                        {stats_items}
                    </div>

                    <!-- Analysis Section -->
                    {analysis_section}

                    <!-- Metadata Section -->
                    <div style="margin-top: 25px;">
                        <h3 style="font-size:15px; color:#2c3e50; border-bottom:2px solid #eee; padding-bottom:5px; margin-bottom:10px;">⚙️ Run Details</h3>
                        <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                            {meta_rows}
                            <tr><td style='padding:6px 10px; border-bottom:1px solid #eee; color:#666;'>Generated On</td><td style='padding:6px 10px; border-bottom:1px solid #eee; font-weight:600;'>{today}</td></tr>
                        </table>
                    </div>

                    <!-- Footer Note -->
                    <div style="margin-top:25px; padding:12px; background:#e3f2fd; border-left:4px solid #2196f3; color:#0d47a1; font-size:13px; border-radius:4px;">
                        ℹ️ <strong>Note:</strong> This is an automated message. Please review the attachment for specific code pointers.
                    </div>
                </div>

                <div style="background: #f8f9fa; padding: 15px; text-align: center; font-size: 11px; color: #95a5a6; border-top: 1px solid #eee;">
                    QGenie Automation Ecosystem • Internal Use Only
                </div>
            </div>
        </body>
        </html>
        """

if __name__ == "__main__":
    reporter = CodebaseEmailReporter()
    
    dummy_stats = {"Files Scanned": 10, "Issues Fixed": 5}
    dummy_meta = {"Project": "TestProject", "User": "pavanr"}
    
    # Test sending (pass "" if no attachment for test)
    reporter.send_report(
        recipients="sendpavanr@gmail.com", 
        attachment_path="", 
        metadata=dummy_meta, 
        stats=dummy_stats, 
        analysis_summary="This is a test run to verify email fallback logic."
    )