package db

import "reflect"

type Table struct {
	name    string
	SQLName string
	Fields  []*Field
}

func (t *Table) SQLOptions() []*Options {
	result := []*Options{}
	for _, f := range t.Fields {
		result := append(result, f.SQL)
	}
	return result
}

func NewTable(any interface{}) (*Table, error) {
	if IsSlice(any) {
		any
	}
}

func IsSlice(any interface{}) bool {
	return ValueOf(any).Kind() == reflect.Slice
}

// ValueOf is reflect.ValueOf but indirect
func ValueOf(any interface{}) reflect.Value {
	return reflect.Indirect(reflect.ValueOf(any))
}
