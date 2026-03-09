class ApprovalRequiredError(Exception):
    def __init__(self, tool_name: str, description: str, category: str = "run_shell"):
        self.tool_name = tool_name
        self.description = description
        self.category = category
        super().__init__(f"Approval required for {tool_name}: {description}")
