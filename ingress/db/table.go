package db

import "reflect"

type Table struct {
	name    string
	SQLName string
	Fields  []*Field
}

func IsSlice(any interface{}) bool {
	return ValueOf(any).Kind() == reflect.Slice
}

// ValueOf is reflect.ValueOf but indirect
func ValueOf(any interface{}) reflect.Value {
	return reflect.Indirect(reflect.ValueOf(any))
}
