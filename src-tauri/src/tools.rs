//! Built-in, app-side tools the chat UI can offer the model and execute on user approval (sc-7772a).
//!
//! Each tool is an OpenAI function: a spec ([`builtin_tool_specs`]) the frontend sends as the chat
//! request's `tools`, plus an executor ([`execute_builtin_tool`]) the `execute_tool` Tauri command
//! runs after the user approves a call. This first set is deliberately **safe** — no network, no
//! filesystem, no secrets — so the tool-calling loop is proven without a security surface. Riskier
//! tools (fetch_url, read_file, web_search, MCP) land as separately-scoped follow-ups.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

/// The OpenAI function-tool definitions for every built-in tool, in the
/// `{"type":"function","function":{name,description,parameters}}` shape the chat endpoint expects.
pub fn builtin_tool_specs() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate an arithmetic expression and return the numeric result. \
                    Supports + - * / %, ^ (power), parentheses, and unary minus.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The arithmetic expression to evaluate, e.g. \"(2 + 3) * 4\"."
                        }
                    },
                    "required": ["expression"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current date and time as an RFC-3339 / ISO-8601 UTC timestamp.",
                "parameters": {"type": "object", "properties": {}}
            }
        }),
    ]
}

/// Execute a built-in tool by name with its JSON argument object, returning the result text that
/// becomes the `tool` message content. An unknown tool or invalid arguments is an `Err` surfaced to
/// the UI (and, if the user re-sends it, to the model as the tool result).
pub fn execute_builtin_tool(name: &str, arguments: &Value) -> Result<String, String> {
    match name {
        "calculator" => {
            let expression = arguments
                .get("expression")
                .and_then(Value::as_str)
                .ok_or_else(|| "calculator requires a string 'expression' argument".to_string())?;
            let value = eval_arithmetic(expression)?;
            Ok(format!("{value}"))
        }
        "get_current_time" => Ok(current_time_rfc3339()),
        other => Err(format!("unknown tool '{other}'")),
    }
}

/// Evaluate a basic arithmetic expression to an `f64`. Grammar (precedence low→high):
///   expr  = term (('+'|'-') term)* ; term = power (('*'|'/'|'%') power)* ;
///   power = unary ('^' power)? (right-assoc) ; unary = ('+'|'-')* primary ;
///   primary = number | '(' expr ')'.
/// Whitespace is ignored; there are no variables, functions, or side effects, so it is safe to run on
/// model output. Division/modulo by zero yields a non-finite value, which is rejected here.
fn eval_arithmetic(input: &str) -> Result<f64, String> {
    let mut parser = Parser {
        bytes: input.as_bytes(),
        pos: 0,
    };
    let value = parser.expr()?;
    parser.skip_ws();
    if parser.pos != parser.bytes.len() {
        return Err(format!(
            "unexpected character '{}' in expression",
            parser.bytes[parser.pos] as char
        ));
    }
    if !value.is_finite() {
        return Err("expression did not evaluate to a finite number".to_string());
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn expr(&mut self) -> Result<f64, String> {
        let mut value = self.term()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'+') => {
                    self.pos += 1;
                    value += self.term()?;
                }
                Some(b'-') => {
                    self.pos += 1;
                    value -= self.term()?;
                }
                _ => break,
            }
        }
        Ok(value)
    }

    fn term(&mut self) -> Result<f64, String> {
        let mut value = self.power()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'*') => {
                    self.pos += 1;
                    value *= self.power()?;
                }
                Some(b'/') => {
                    self.pos += 1;
                    value /= self.power()?;
                }
                Some(b'%') => {
                    self.pos += 1;
                    value %= self.power()?;
                }
                _ => break,
            }
        }
        Ok(value)
    }

    fn power(&mut self) -> Result<f64, String> {
        let base = self.unary()?;
        self.skip_ws();
        if self.peek() == Some(b'^') {
            self.pos += 1;
            let exponent = self.power()?; // right-associative
            Ok(base.powf(exponent))
        } else {
            Ok(base)
        }
    }

    fn unary(&mut self) -> Result<f64, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'+') => {
                self.pos += 1;
                self.unary()
            }
            Some(b'-') => {
                self.pos += 1;
                Ok(-self.unary()?)
            }
            _ => self.primary(),
        }
    }

    fn primary(&mut self) -> Result<f64, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'(') => {
                self.pos += 1;
                let value = self.expr()?;
                self.skip_ws();
                if self.peek() != Some(b')') {
                    return Err("missing closing parenthesis".to_string());
                }
                self.pos += 1;
                Ok(value)
            }
            Some(c) if c.is_ascii_digit() || c == b'.' => self.number(),
            Some(c) => Err(format!(
                "unexpected character '{}' in expression",
                c as char
            )),
            None => Err("unexpected end of expression".to_string()),
        }
    }

    fn number(&mut self) -> Result<f64, String> {
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit() || c == b'.') {
            self.pos += 1;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| "invalid number".to_string())?;
        text.parse::<f64>()
            .map_err(|_| format!("invalid number '{text}'"))
    }
}

/// The current time as an RFC-3339 UTC timestamp (`YYYY-MM-DDTHH:MM:SSZ`).
fn current_time_rfc3339() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    format_rfc3339_utc(secs)
}

/// Format Unix epoch seconds as an RFC-3339 UTC timestamp via Howard Hinnant's civil-from-days
/// algorithm — keeps the contract free of a date dependency.
fn format_rfc3339_utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (hour, minute, second) = (rem / 3600, (rem % 3600) / 60, rem % 60);

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };

    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_builtin_tools_as_openai_functions() {
        let specs = builtin_tool_specs();
        let names: Vec<&str> = specs
            .iter()
            .map(|spec| spec["function"]["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"calculator"));
        assert!(names.contains(&"get_current_time"));
        assert_eq!(specs[0]["type"], "function");
    }

    #[test]
    fn calculator_respects_precedence_and_parens() {
        assert_eq!(eval_arithmetic("2 + 3 * 4").unwrap(), 14.0);
        assert_eq!(eval_arithmetic("(2 + 3) * 4").unwrap(), 20.0);
        assert_eq!(eval_arithmetic("2 ^ 3 ^ 2").unwrap(), 512.0); // right-associative
        assert_eq!(eval_arithmetic("-3 + 7").unwrap(), 4.0);
        assert_eq!(eval_arithmetic("10 % 3").unwrap(), 1.0);
        assert_eq!(eval_arithmetic("1.5e2 / 3").unwrap(), 50.0);
    }

    #[test]
    fn calculator_executes_via_dispatch() {
        let out =
            execute_builtin_tool("calculator", &json!({"expression": "(2 + 3) * 4"})).unwrap();
        assert_eq!(out, "20");
    }

    #[test]
    fn calculator_rejects_garbage_and_div_by_zero() {
        assert!(execute_builtin_tool("calculator", &json!({"expression": "2 +"})).is_err());
        assert!(execute_builtin_tool("calculator", &json!({"expression": "1 / 0"})).is_err());
        assert!(execute_builtin_tool("calculator", &json!({"expression": "rm -rf /"})).is_err());
        // Missing / wrong-typed argument is an error, not a panic.
        assert!(execute_builtin_tool("calculator", &json!({})).is_err());
    }

    #[test]
    fn unknown_tool_is_an_error() {
        assert!(execute_builtin_tool("definitely_not_a_tool", &json!({})).is_err());
    }

    #[test]
    fn formats_epoch_as_rfc3339_utc() {
        assert_eq!(format_rfc3339_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(format_rfc3339_utc(1_700_000_000), "2023-11-14T22:13:20Z");
    }

    #[test]
    fn get_current_time_round_trips_through_dispatch() {
        let out = execute_builtin_tool("get_current_time", &json!({})).unwrap();
        // Shape check: "YYYY-MM-DDTHH:MM:SSZ".
        assert_eq!(out.len(), 20);
        assert!(out.ends_with('Z'));
        assert_eq!(&out[4..5], "-");
    }
}
