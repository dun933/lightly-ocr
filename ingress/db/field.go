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
	fields, err := CollectFields(st, []*Field{})
	if err != nil {
		return nil, err
	}
	return fields, nil
}

func CollectFields(st interface{}, fields []*Field) ([]*Field, error) {
	iter := NewFieldIteration(st)
	for iter.Next() {
		if iter.IsEmbeddedStruct() {
			subField, err := CollectFields(iter.ValueField().Interface(), fields)
			if err != nil {
				return nil, err
			}
			fields = subField
			continue
		}

		sqlOptions, err := iter.SQLOptions()

		if err != nil {
			return nil, err
		}
		fields = append(fields, &Field{
			Name:  iter.Name(),
			Value: iter.Value(),
			SQL:   sqlOptions,
		})
	}
	return fields, nil
}

func NewFieldIteration(st interface{}) *FieldIteration {
	rValue := reflect.Indirect(reflect.ValueOf(st))
	rType := rValue.Type()
	length := rValue.NumField()
	return &FieldIteration{
		Index:        -1,
		Length:       length,
		ReflectValue: rValue,
		ReflectType:  rType,
	}
}

func (it *FieldIteration) Next() bool {
	if it.Index+1 >= it.Length {
		return false
	}
	it.Index++
	return true
}

func (it *FieldIteration) TypeField() reflect.StructField {
	return it.ReflectType.Field(it.Index)
}

func (it *FieldIteration) ValueField() reflect.Value {
	return it.ReflectValue.Field(it.Index)
}

func (it *FieldIteration) IsEmbeddedStruct() bool {
	if _, ok := TypeDict[it.TypeField().Type.String()]; ok {
		return false
	}
	return it.ReflectValue.Field(it.Index).Kind() == reflect.Struct
}
