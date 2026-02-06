import openpyxl
import os
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from pathlib import Path

class ExcelWriter:
    """
    Generic Excel writer utility for creating, formatting, and saving Excel reports
    with metadata and result tables. 
    Primary library: openpyxl.
    """
    def __init__(self, file_path, env_config):
        """
        Initializes a new Excel workbook and sets the output file path.

        Args:
            file_path (str or Path): Destination path for the generated Excel file.
        """
        self.file_path = Path(file_path)
        self.wb = openpyxl.Workbook()
        self._init_workbook()
        self.env = env_config
       
    def is_pass(self, value):
        """Check if value matches any 'pass' label (case-insensitive)."""
        return str(value).strip().upper() in self.PASS_LABELS

    def is_fail(self, value):
        """Check if value matches any 'fail' label (case-insensitive)."""
        return str(value).strip().upper() in self.FAIL_LABELS

    def _init_workbook(self):
        """
        Removes the default worksheet created by openpyxl (typically named 'Sheet').
        Ensures all sheets are user-defined.
        """
        if 'Sheet' in self.wb.sheetnames:
            std = self.wb['Sheet']
            self.wb.remove(std)

    def add_data_sheet(self, data_dict, title, report_title):
        """
        Adds a metadata summary sheet showing key-value pairs.
        Arranges a title row and two columns: Parameter and Value.

        Args:
            data_dict (dict): Mapping of metadata parameter names to values.
            title (str): Name for the worksheet/tab in Excel.
            report_title (str): Main title displayed at the top of the sheet.
        """
        ws = self.wb.create_sheet(title)

        # Merge top row for main report title, style for emphasis
        ws.merge_cells('A1:B1')
        ws['A1'] = report_title
        ws['A1'].font = Font(size=16, bold=True)
        ws['A1'].alignment = Alignment(horizontal="center")

        # Write column headers for metadata table
        ws['A2'] = "Parameter"
        ws['B2'] = "Value"
        ws['A2'].font = ws['B2'].font = Font(bold=True)
        ws['A2'].fill = ws['B2'].fill = PatternFill("solid", fgColor="E7E6E6")

        # Set reasonable column widths
        ws.column_dimensions['A'].width = 20
        ws.column_dimensions['B'].width = 64

        # Populate metadata rows
        for i, (k, v) in enumerate(data_dict.items(), start=3):
            ws[f'A{i}'], ws[f'B{i}'] = k, v
            ws[f'A{i}'].font = Font(bold=True)
            ws[f'A{i}'].fill = PatternFill("solid", fgColor="E7E6E6")
            ws[f'B{i}'].fill = PatternFill("solid", fgColor="FFFFFF")  # White
            ws[f'A{i}'].alignment = Alignment(horizontal="right")

    def add_table_sheet(
        self, 
        headers, 
        data_rows, 
        sheet_name="Results", 
        conditional_formats=None,
        autofit=True
    ):
        """
        Adds a data table sheet, e.g. for test case results, with formatted headers and
        optional per-cell conditional formatting.

        Args:
            headers (List[str]): List of column header names.
            data_rows (List[List]): Row data, each corresponding to a header.
            sheet_name (str): Worksheet name for this data table.
            conditional_formats (callable): Function(cell, col_name, cell_value, row_idx)
                - Used to apply custom cell styles/conditional formatting.
            autofit (bool): If True, column widths are automatically fit to content.
        """
        ws = self.wb.create_sheet(sheet_name)

        # Write and format the header row
        for col, header in enumerate(headers, 1):
            cell = ws[f"{get_column_letter(col)}1"]
            cell.value = header
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="4F81BD")  # Blue header
            cell.alignment = Alignment(horizontal="center")
        
        # Use a uniform thin border for all cells if desired
        thin = Side(border_style="thin", color="5c5c5c")

        # Write data rows, applying optional conditional formatting as needed
        for r, row in enumerate(data_rows, 2):
            if isinstance(row, dict):
                row = [row.get(h, "") for h in headers]
            # Align row length to header count (trim or pad if needed)
            row = list(row[:len(headers)])  
            for c, val in enumerate(row, 1):
                cell = ws.cell(row=r, column=c, value=val)
                # Set cell borders
                cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)
                col_name = headers[c-1].strip().lower()
                cell_value = str(val).strip().upper()
                # Call conditional_formats callback for per-cell styling
                if conditional_formats:                    
                    conditional_formats(cell, col_name, cell_value, r)
                # Default: gray fill for every other row (if no conditional_format)
                elif r % 2 == 0:
                    cell.fill = PatternFill("solid", fgColor="F3F3F3")

        # Automatically adjust column widths to fit contents (optional)
        if autofit:
            for col in ws.columns:
                max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col)
                ws.column_dimensions[col[0].column_letter].width = max(max_len + 2, 14)

    # Example conditional formatting function
    def excel_conditional_formats(self, cell, col_name, cell_value, row_idx, format_header, pass_label, fail_label, env):
        """
        Uses env-configured status labels and colors for formatting.
        """
        col_name = col_name.strip().lower()
        cell_value = str(cell_value).strip().upper()
        pass_label = str(pass_label).strip().upper()
        fail_label = str(fail_label).strip().upper()
        format_header = str(format_header).strip().lower()    
        if (format_header) in col_name:        
            if cell_value in pass_label:
                cell.fill = PatternFill("solid", fgColor=env.get("PASS_COLOR", "C6EFCE"))
            elif cell_value in fail_label:
                cell.fill = PatternFill("solid", fgColor=env.get("FAIL_COLOR", "FFC7CE"))   
        elif row_idx % 2 == 0:
            cell.fill = PatternFill("solid", fgColor=env.get("ALT_ROW_COLOR", "F3F3F3"))
            
    def save(self):
        """
        Saves the workbook to self.file_path. Creates folders if they do not exist.
        """
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.wb.save(str(self.file_path))



