package db

import (
	"fmt"
	"strings"
)

// Types contains default SQL values
var Types = map[string]int{
	"int":     11,
	"bigint":  20,
	"varchar": 255,
}

// TypeDict defines Go types with corresponding SQL types
var TypeDict = map[string]string{
	"float32":         "float",
	"float64":         "float",
	"int":             "int",
	"uint":            "int",
	"int64":           "bigint",
	"uint64":          "bigint",
	"string":          "varchar",
	"bool":            "tinyint",
	"sql.NullFloat32": "float",
	"sql.NullFloat64": "float",
	"sql.NullInt64":   "bigint",
	"sql.NullString":  "varchar",
	"sql.NullBool":    "tinyint",
}

// Options contains placeholder for values included in queries
type Options struct {
	Name   string
	Type   string
	Length int
}

// NewTableQuery handles new table query string
func NewTableQuery(name string, fields []*Options, ifNotExists bool) string {
	NotExistsQuery := ""
	if ifNotExists {
		NotExistsQuery = "IF NOT EXISTS"
	}
	return fmt.Sprintf("CREATE TABLE %s `%s` (\n%s\n);", NotExistsQuery, name, FieldQueries(fields))
}

// DropTableQuery returns a drop query
func DropTableQuery(name string, ifExists bool) string {
	ExistsQuery := ""
	if ifExists {
		ExistsQuery = "IF EXISTS"
	}
	return fmt.Sprintf("DROP TABLE %s %s", ExistsQuery, name)
}

// FieldQueries chains multiple FieldQuery into a string
func FieldQueries(fields []*Options) string {
	queries := []string{}
	for _, f := range fields {
		queries = append(queries, FieldQuery(f))
	}
	return strings.Join(queries, ",\n")
}

// FieldQuery returns a string containing *Options
func FieldQuery(field *Options) string {
	length := ""
	if field.Length > -1 {
		length = fmt.Sprintf("(%d)", field.Length)
	}
	return fmt.Sprintf(" `%s` %s%s", field.Name, field.Type, length)

}

// SelectQuery handles select query
func SelectQuery(tableName string, columnNames []string) string {
	columns := strings.Join(columnNames, ",")
	if columns == "" {
		columns = "*"
	}
	return fmt.Sprintf("SELECT %s FROM %s", columns, tableName)
}

// DeleteQuery returns delete query
func DeleteQuery(tableName, index string) string {
	return fmt.Sprintf("DELETE FROM %s WHERE %s=?", tableName, index)
}

// InsertQuery handles insert query
func InsertQuery(tableName string, columnNames []string) string {
	var questionMarks string

	if len(columnNames) > 0 {
		questionMarks = strings.Repeat("?,", len(columnNames))
		questionMarks = questionMarks[:len(questionMarks)-1]
	}
	return fmt.Sprintf("INSERT INTO %s (%s) VALUES (%s)", tableName, strings.Join(quoteColumnNames(columnNames), ","), questionMarks)
}

// UpdateQuery returns update query
func UpdateQuery(tableName, index string, columnNames []string) string {
	return fmt.Sprintf("UPDATE %s SET %s=? WHERE %s=?", tableName, strings.Join(quoteColumnNames(columnNames), "=?,"), index)
}

// MatchType helps to match Go type with SQL type
func MatchType(typeName string) (string, error) {
	if result, ok := TypeDict[typeName]; ok {
		return result, nil
	}
	return "", fmt.Errorf(fmt.Sprintf("cannot match Go type '%s' with SQL type", typeName))
}

// quoteColumnNames returns all columns with quotes (string formats)
func quoteColumnNames(columns []string) []string {
	quoted := []string{}

	for _, c := range columns {
		quoted = append(quoted, fmt.Sprintf("`%s`", c))
	}
	return quoted
}
