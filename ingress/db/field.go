package db

import "reflect"

type Field struct {
	Name  string
	Value interface{}
	SQL   *Options
}

type FieldIteration struct {
	Index        int
	Length       int
	ReflectValue reflect.Value
	ReflectType  reflect.Type
}

func GetFieldsOf(st interface{}) ([]*Field, error) {
	fields, err := GetFields(st, []*Field{})
	if err != nil {
		return nil, err
	}
	return fields, nil
}

func GetFields(st interface{}, fields []*Field) ([]*Field, error) {
	iter := NewFieldIteration(st)
	for iter.Next() {
		if iter.IsEmbeddedStruct() {
			if _fields, err := GetFields(iter.ValueField().Interface(), fields); err != nil {
				return nil, err
			} else {
				fields = _fields
			}
			continue
		}

		sqlOptions, err := iter.SQLOptions()

		if err != nil {
			return nil, err
		}
	}
}
