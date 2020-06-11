package db

type RowValue struct {
	SQLColumn string
	Value     interface{}
}

type Row struct {
	SQLTableName string
	Values       []*RowValue
}

func (r *Row) SQLValues() map[string]interface{} {
	result := map[string]interface{}{}
	for _, v := range r.Values {
		result[v.SQLColumn] = v.Value
	}

	return result
}

func CreateRow(st interface{}) (*Row, error) {
	values, err := Get
}
